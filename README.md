# A-Z Indian Medicine Dataset Usecase Enrichment

A Python-based solution for enhancing a comprehensive Indian medicine dataset with specific disease and symptom information using the Google Gemini 2.0 Flash API.

## 📋 Project Overview

This project addresses the challenge of enriching a large dataset (200,000+ medicines) with standardized medical usecase information. By leveraging Google's Gemini AI, the system automatically identifies and adds the specific symptoms and diseases each medicine treats, transforming raw pharmaceutical data into a more valuable and searchable resource. (may contain missing values)

## 🌟 Features

- **AI-Powered Analysis**: Uses Gemini 2.0 Flash API to analyze medicine names and compositions
- **Precise Disease/Symptom Focus**: Extracts specific medical conditions rather than general descriptions
- **Rate-Limit Handling**: Sophisticated backoff strategies and error recovery for API quotas
- **Hybrid Approach**: Combines AI lookups with a pre-defined knowledge base of common medicines
- **Resumable Processing**: Can pause and continue processing where it left off
- **Comprehensive Logging**: Detailed tracking of each processing step and error
- **Data Validation**: Multi-step cleaning and verification to ensure quality results

## 🔧 Technical Implementation

### Download dataset
Kaggle page: https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india
```bash
curl -L -o ~/Downloads/az-medicine-dataset-of-india.zip\
  https://www.kaggle.com/api/v1/datasets/download/shudhanshusingh/az-medicine-dataset-of-india
```

### Prerequisites

```bash
pip install pandas google-generativeai tqdm backoff
```

### Configuration

1. Obtain a Google API key for Gemini from [Google AI Studio](https://makersuite.google.com/)
2. Replace `YOUR_GOOGLE_API_KEY` in the script with your actual API key
3. Adjust batch sizes and wait periods based on your API quota limits

### Input Dataset Format

The script expects a CSV file with at least the following columns:
- `id`
- `name` (medicine name)
- `price`
- `Is_discontinued`
- `manufacturer_name`
- `type`
- `pack_size_label`
- `short_composition1` (main active ingredient)
- `short_composition2` (secondary ingredient if available)

### Output

The script adds a `Usecase` column containing standardized, comma-separated symptoms and diseases:

| name | short_composition1 | Usecase |
|------|-------------------|---------|
| Augmentin 625 Duo Tablet | Amoxycillin (500mg) | bacterial infections, sinusitis, pneumonia |
| Azithral 500 Tablet | Azithromycin (500mg) | bacterial infections, respiratory infections |

## 📊 Processing Approach

The script implements a two-pass approach:

1. **First Pass**: Uses a pre-populated lookup table to quickly assign usecases to common medicines
2. **Second Pass**: Processes remaining medicines using the Gemini API with:
   - Pharmacology-focused prompting
   - Post-processing to standardize format
   - Validation to ensure quality
   - Adaptive rate-limit handling

## 🚀 Usage

### Basic Usage

```bash
# Update input_file in the script to point to your dataset
python medicine_usecase_processor.py
```

### Processing Large Datasets

For datasets with 200,000+ entries:

1. Consider dividing your dataset into smaller chunks
2. Expand the `COMMON_MEDICINE_USECASES` dictionary with more common medicines
3. Adjust batch size and delay parameters based on observed rate limiting

## 📝 Logging and Monitoring

The script generates a detailed log file (`medicine_usecase_progress.log`) tracking:
- Processing start and completion times
- Rate limit events
- Processing errors
- Successfully processed medicines

## 🛠️ Customization

### Adjusting AI Prompt

The `get_medicine_usecase()` function contains the prompt template sent to Gemini. You can customize this to focus on specific aspects of medicine usecases.

### Expanding the Lookup Database

The `COMMON_MEDICINE_USECASES` dictionary can be expanded with additional medicines to reduce API calls.

## ⚠️ Limitations

- API rate limits may restrict processing speed
- Some uncommon medicines may not get accurate usecases
- Processing 200,000+ medicines may take several days depending on rate limits

## 📈 Future Enhancements

- Integration with offline medical databases
- Support for multiple language outputs
- Categorization of usecases by medical specialty
- Addition of contraindications and side effects
- Implementation of medicine similarity detection

## 📚 Additional Resources

- [Google Generative AI Python SDK](https://github.com/google/generative-ai-python)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Backoff Library Documentation](https://github.com/litl/backoff)

## 🙏 Acknowledgements

This project builds upon the comprehensive A-Z Medicine dataset of India and utilizes Google's Gemini AI capabilities for natural language understanding of pharmaceutical compositions.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
