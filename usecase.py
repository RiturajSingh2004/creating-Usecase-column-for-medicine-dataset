import pandas as pd
import os
import time
import google.generativeai as genai
from tqdm import tqdm
import backoff
import re
import random
from datetime import datetime

# Configure Google Gemini API
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"  # Replace with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# Define the model
model = genai.GenerativeModel('gemini-2.0-flash-001')

# Function to clean and standardize usecase text
def clean_usecase(text):
    """Clean and standardize usecase text to contain only symptoms/diseases"""
    # Remove phrases like "used for", "treatment of", etc.
    text = re.sub(r'used (in|for|to)|treatment of|indicated for|helps with|treats', '', text, flags=re.IGNORECASE)
    
    # Remove other explanatory phrases
    text = re.sub(r'(such as|including|like|e\.g\.|i\.e\.)', '', text, flags=re.IGNORECASE)
    
    # Remove any text in parentheses
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove any text that starts with "it" or "this"
    text = re.sub(r'(it|this)( is| can| may| will)?( be| also)? [^,]*,?', '', text, flags=re.IGNORECASE)
    
    # Split by commas and clean each item
    items = [item.strip() for item in text.split(',')]
    
    # Remove any items that are too long (likely sentences)
    items = [item for item in items if len(item) < 50 and len(item) > 0]
    
    # Join back with commas
    return ', '.join(items)

# Special handler for rate limit errors
def is_rate_limit_error(exception):
    """Check if the exception is due to rate limiting"""
    return "429" in str(exception) or "quota" in str(exception).lower() or "resource exhausted" in str(exception).lower()

# Function to generate usecase with backoff for rate limiting
@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=3,
    giveup=lambda e: not is_rate_limit_error(e)  # Only retry for rate limit errors
)
def get_medicine_usecase(row):
    """Get usecase for a medicine using Gemini API with proper prompt engineering"""
    try:
        # Create a structured prompt focused on specific symptoms/diseases
        prompt = f"""
        As a pharmacologist, list ONLY the specific symptoms or diseases treated by this medicine:
        
        Medicine Name: {row['name']}
        Active Components: {row['short_composition1']} {row['short_composition2'] if pd.notna(row['short_composition2']) else ''}
        Medicine Type: {row['type']}
        
        Format: Provide ONLY a comma-separated list of specific symptoms or diseases.
        Example good response: "fever, headache, common cold"
        Example bad response: "This medicine is used for treatment of fever and related symptoms."
        
        DO NOT write sentences or phrases like "used for", "treats", etc.
        DO NOT provide general categories like "pain relief" - be specific like "headache, joint pain"
        Keep each symptom or disease to 1-3 words when possible.
        """
        
        # Get completion from Gemini
        response = model.generate_content(prompt)
        usecase = response.text.strip().strip('"\'')
        
        # Clean and standardize the response
        usecase = clean_usecase(usecase)
        
        # If after cleaning it's empty, use a fallback
        if not usecase:
            return "unknown"
            
        return usecase
    
    except Exception as e:
        if is_rate_limit_error(e):
            print(f"\nRate limit exceeded at {datetime.now().strftime('%H:%M:%S')}. Waiting before retrying...")
            # Wait for a longer period before retrying - exponential backoff will handle this
            raise e  # Re-raise to let backoff handle it
        else:
            print(f"Error processing {row['name']}: {e}")
            return "unknown"

# Function to validate usecase format
def validate_usecase(usecase):
    """Check if the usecase follows the desired format"""
    # Check if it's too long (likely a sentence)
    if len(usecase) > 150:
        return False
        
    # Check if it contains verbs that suggest sentences
    sentence_patterns = [
        r'\b(is|are|was|were|will|should|could|would|can|may|might|must|has|have|had|does|do|did)\b',
        r'\b(treat|use|help|provide|reduce|prevent|manage|relieve|alleviate)\b'
    ]
    
    for pattern in sentence_patterns:
        if re.search(pattern, usecase, re.IGNORECASE):
            return False
            
    return True

def process_dataset(input_file, output_file, batch_size=50):
    """Process the medicine dataset in batches to add usecase field"""
    # Read the CSV file
    print(f"Reading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Check if Usecase column already exists
    if 'Usecase' not in df.columns:
        df['Usecase'] = None
    
    total_rows = len(df)
    print(f"Total medicines in dataset: {total_rows}")
    
    # Create a temporary dict to store results
    results_dict = {}
    
    # Create a log file to track progress
    log_file = "medicine_usecase_progress.log"
    with open(log_file, "a") as log:
        log.write(f"\n--- Processing started at {datetime.now()} ---\n")
    
    # Process in batches
    batch_count = (total_rows + batch_size - 1) // batch_size
    
    for i in tqdm(range(batch_count), desc="Processing batches"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_rows)
        
        batch = df.iloc[start_idx:end_idx]
        rows_to_process = batch[batch['Usecase'].isna() | (batch['Usecase'] == '') | (batch['Usecase'] == 'unknown')].index
        
        if len(rows_to_process) == 0:
            continue
        
        # Log batch start
        with open(log_file, "a") as log:
            log.write(f"Starting batch {i+1}/{batch_count} at {datetime.now()}\n")
            
        # Add randomized delay between batches to avoid predictable patterns
        delay = random.uniform(1.5, 5.0)
        time.sleep(delay)
        
        for idx in tqdm(rows_to_process, desc=f"Batch {i+1}/{batch_count}", leave=False):
            row = df.loc[idx]
            
            try:
                # Try to get usecase
                usecase = get_medicine_usecase(row)
                
                # Additional validation
                if not validate_usecase(usecase):
                    # If validation fails, try to clean it again
                    usecase = clean_usecase(usecase)
                    if not validate_usecase(usecase):
                        usecase = "unknown"
                
                # Store in dictionary
                results_dict[idx] = usecase
                
                # Log successful processing
                with open(log_file, "a") as log:
                    log.write(f"Processed {row['name']}: {usecase}\n")
                
                # Variable delay between requests
                time.sleep(random.uniform(0.5, 2.0))
                
            except Exception as e:
                if is_rate_limit_error(e):
                    # If we get a rate limit error, save progress and wait
                    print(f"\nRate limit hit. Saving progress and waiting...")
                    with open(log_file, "a") as log:
                        log.write(f"Rate limit hit at {datetime.now()}, saving progress\n")
                    
                    # Save current progress
                    for k, v in results_dict.items():
                        df.at[k, 'Usecase'] = v
                    df.to_csv(output_file, index=False)
                    
                    # Wait for a longer time before continuing
                    long_wait = random.uniform(60, 120)  # Wait 1-2 minutes
                    print(f"Waiting for {long_wait:.1f} seconds before resuming...")
                    time.sleep(long_wait)
                    
                    # Clear results dict after saving
                    results_dict = {}
                else:
                    # For other errors, log and continue
                    print(f"Error processing {row['name']}: {e}")
                    with open(log_file, "a") as log:
                        log.write(f"Error processing {row['name']}: {e}\n")
            
            # Save progress periodically
            if len(results_dict) >= 20:
                # Apply updates to dataframe
                for k, v in results_dict.items():
                    df.at[k, 'Usecase'] = v
                
                # Save to file
                df.to_csv(output_file, index=False)
                print(f"Saved progress to {output_file}")
                
                # Clear results dict after saving
                results_dict = {}
        
        # At the end of each batch, save any remaining results
        if results_dict:
            for k, v in results_dict.items():
                df.at[k, 'Usecase'] = v
            results_dict = {}
            
        # Save after each batch
        df.to_csv(output_file, index=False)
        print(f"Completed batch {i+1}/{batch_count}, saved to {output_file}")
        
        # Log batch completion
        with open(log_file, "a") as log:
            log.write(f"Completed batch {i+1}/{batch_count} at {datetime.now()}\n")
    
    print(f"Processing complete. Enhanced dataset saved to {output_file}")
    return df

# Common medicines lookup for fallback
COMMON_MEDICINE_USECASES = {
    "Augmentin": "bacterial infections, sinusitis, pneumonia, ear infections",
    "Azithromycin": "bacterial infections, respiratory infections, skin infections",
    "Amoxycillin": "bacterial infections, bronchitis, pneumonia, tonsillitis",
    "Paracetamol": "fever, pain, headache",
    "Ibuprofen": "pain, inflammation, fever, arthritis",
    "Montelukast": "asthma, allergic rhinitis",
    "Fexofenadine": "allergies, hay fever, urticaria",
    "Cetirizine": "allergies, hay fever, urticaria",
    "Hydroxyzine": "anxiety, itching, allergies",
    "Levosalbutamol": "asthma, bronchospasm, COPD",
    "Ambroxol": "cough, bronchitis, respiratory congestion",
    "Pheniramine": "allergies, hay fever, itching",
    "Clavulanic Acid": "bacterial infections"
}

def get_usecase_from_lookup(medicine_name, compositions):
    """Try to get usecase from lookup table based on medicine name or compositions"""
    # Check if the medicine name contains any key from the lookup
    for key in COMMON_MEDICINE_USECASES:
        if key.lower() in medicine_name.lower():
            return COMMON_MEDICINE_USECASES[key]
    
    # Check if any composition contains a key from the lookup
    if compositions:
        for comp in compositions:
            if pd.notna(comp) and comp:
                # Extract the first word (active ingredient)
                active = comp.split()[0] if comp.split() else ""
                if active.lower() in COMMON_MEDICINE_USECASES:
                    return COMMON_MEDICINE_USECASES[active.lower()]
    
    return None

def process_dataset_with_fallback(input_file, output_file, batch_size=50):
    """Process the dataset with fallback to common medicines lookup"""
    # Read the CSV file
    print(f"Reading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Check if Usecase column already exists
    if 'Usecase' not in df.columns:
        df['Usecase'] = None
    
    # First pass: try to fill using the lookup table to reduce API calls
    print("First pass: Using lookup table for common medicines...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        if pd.isna(df.at[idx, 'Usecase']) or df.at[idx, 'Usecase'] == '' or df.at[idx, 'Usecase'] == 'unknown':
            usecase = get_usecase_from_lookup(row['name'], [row['short_composition1'], row['short_composition2']])
            if usecase:
                df.at[idx, 'Usecase'] = usecase
    
    # Save progress after first pass
    df.to_csv(output_file, index=False)
    print(f"First pass complete. Saved to {output_file}")
    
    # Second pass: use API for remaining medicines
    print("Second pass: Using API for remaining medicines...")
    df = process_dataset(output_file, output_file, batch_size)
    
    return df

if __name__ == "__main__":
    # File paths
    input_file = "A_Z_medicines_dataset_of_India.csv"  # Update with your input file name
    output_file = "a_z_medicines_with_usecases.csv"
    
    # Check if output file exists to enable resuming
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Resuming from last saved state.")
        input_file = output_file
    
    # Process the dataset with fallback mechanism
    enhanced_df = process_dataset_with_fallback(input_file, output_file, batch_size=50)
    
    # Print sample of processed data
    print("\nSample of processed data:")
    print(enhanced_df[['name', 'short_composition1', 'Usecase']].head(10))
    
    # Print statistics
    null_count = enhanced_df['Usecase'].isna().sum()
    unknown_count = (enhanced_df['Usecase'] == 'unknown').sum()
    print(f"\nStatistics:")
    print(f"Total medicines: {len(enhanced_df)}")
    print(f"Medicines with usecases: {len(enhanced_df) - null_count - unknown_count}")
    print(f"Medicines with unknown usecases: {unknown_count}")
    print(f"Medicines with missing usecases: {null_count}")