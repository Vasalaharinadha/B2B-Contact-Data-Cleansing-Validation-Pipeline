# B2B-Contact-Data-Cleansing-Validation-Pipeline



A comprehensive data engineering solution for cleaning, validating, and standardizing B2B contact information - perfect for sales intelligence platforms.

## 🎯 Project Overview

This pipeline processes raw B2B contact data and transforms it into high-quality, verified records suitable for sales prospecting and CRM integration. Built with pandas and data engineering best practices.

## ✨ Key Features

### Data Cleaning & Validation
- **Email Validation**: Regex-based validation, domain extraction, business vs. personal email detection
- **Phone Number Standardization**: E.164 format conversion, international format handling
- **Duplicate Detection**: Multi-column deduplication strategies
- **Data Quality Scoring**: Automated quality assessment for each record

### Data Standardization
- **Job Title Normalization**: Seniority level categorization (C-Level, VP, Director, Manager, etc.)
- **Department Classification**: Automatic department assignment (Sales, Marketing, Engineering, etc.)
- **Company Name Standardization**: Remove legal suffixes (Inc., LLC, Ltd., Corp.)
- **Data Enrichment**: Completeness metrics, quality scores, timestamps

### Pipeline Features
- **Modular Architecture**: Each transformation is a separate, testable component
- **Comprehensive Logging**: Track pipeline execution and data quality metrics
- **Multiple Format Support**: CSV, Excel, JSON input/output
- **Statistics Dashboard**: Real-time pipeline performance metrics

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/b2b-contact-data-pipeline.git
cd b2b-contact-data-pipeline

# Install required packages
pip install -r requirements.txt
```

## 📦 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
openpyxl>=3.0.0
```

## 💻 Usage

### Quick Start

```python
from contact_data_pipeline import B2BContactDataPipeline

# Initialize pipeline
pipeline = B2BContactDataPipeline()

# Run complete pipeline
cleaned_data = pipeline.run_pipeline(
    input_file='raw_contacts.csv',
    output_file='cleaned_contacts.csv'
)
```

### Generate Sample Data

```python
from contact_data_pipeline import generate_sample_data

# Generate 1000 sample B2B contacts
generate_sample_data('sample_contacts.csv', num_records=1000)
```

### Individual Components

```python
import pandas as pd
from contact_data_pipeline import B2BContactDataPipeline

pipeline = B2BContactDataPipeline()
df = pd.read_csv('contacts.csv')

# Clean emails only
df = pipeline.clean_email_addresses(df)

# Standardize phone numbers
df = pipeline.clean_phone_numbers(df)

# Categorize job titles
df = pipeline.standardize_job_titles(df)

# Standardize company names
df = pipeline.standardize_company_names(df)
```

## 📊 Output Fields

The pipeline enriches your data with these additional fields:

| Field | Description |
|-------|-------------|
| `email_valid` | Boolean flag for email validity |
| `email_domain` | Extracted email domain |
| `is_business_email` | Flag for business vs. personal email |
| `phone_cleaned` | Standardized phone in E.164 format |
| `phone_valid` | Boolean flag for phone validity |
| `seniority_level` | Categorized seniority (C-Level, VP, Director, etc.) |
| `department` | Categorized department (Sales, Marketing, Engineering, etc.) |
| `company_name_cleaned` | Standardized company name |
| `data_quality_score` | Overall quality score (0-100) |
| `data_completeness` | Percentage of non-null fields |
| `processed_timestamp` | Pipeline execution timestamp |

## 📈 Data Quality Scoring

Quality scores are calculated based on:
- Valid email address: 30 points
- Valid phone number: 25 points
- Business email (not personal): 20 points
- Company name present: 15 points
- Job title categorized: 10 points

**Total possible score: 100 points**

Records with scores ≥50 are considered high-quality.

## 🎯 Use Cases

Perfect for:
- **Sales Intelligence Platforms**: Clean and verify B2B contact databases
- **CRM Data Migration**: Standardize data before importing to Salesforce, HubSpot, etc.
- **Lead Enrichment**: Enhance existing contact lists with quality metrics
- **Data Deduplication**: Remove duplicate contacts across multiple sources
- **GTM Operations**: Prepare verified contact lists for outbound campaigns

## 📊 Sample Output

```
============================================================
B2B CONTACT DATA PIPELINE - STATISTICS
============================================================
Total Records Processed: 1000
High Quality Records: 823
Low Quality Records: 177
Duplicates Removed: 47
============================================================
```

## 🔧 Technical Details

### Architecture
- **Object-Oriented Design**: Modular, maintainable pipeline class
- **Type Hints**: Full type annotations for better code clarity
- **Error Handling**: Comprehensive exception handling and logging
- **Scalability**: Efficient pandas operations for large datasets

### Data Processing Flow
1. **Load** → Read data from CSV/Excel/JSON
2. **Clean** → Email and phone validation
3. **Standardize** → Job titles and company names
4. **Deduplicate** → Remove duplicates
5. **Enrich** → Add quality scores and metadata
6. **Export** → Save cleaned data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Your Name**
- Portfolio: [your-portfolio.com]
- LinkedIn: [your-linkedin]
- GitHub: [@yourusername]

## 🙏 Acknowledgments

Built as a portfolio project demonstrating data engineering expertise for B2B sales intelligence platforms like Zintlr.

---

⭐ If you find this project useful, please consider giving it a star!
