# Data Download and Preparation Guide

This document provides instructions for downloading and preparing the agricultural query dataset for further application training and preprocessing. 
**Note:** This dataset is not required for the initial setup of the AgriChat-Annam application, but will be used in subsequent steps for model fine-tuning, data augmentation, or analytics.

---

## Dataset Description

The dataset contains real-world agricultural queries and responses, with rich metadata for each entry. Each row includes:

- **BlockName**: Block or region code/name
- **Category**: Crop or product category (e.g., Fruits, Vegetables)
- **Year**: Year of the query
- **Month**: Month of the year of the query (e.g., 1, 2, ..., 12)
- **Day**: Day of the month (1-31)
- **Crop**: Name of the crop (e.g., capsicum, beans)
- **DistrictName**: District or locality name
- **QueryType**: Type of user query (e.g., fertilizer use, plant protection)
- **Season**: Agricultural season (e.g., RABI, KHARIF)
- **Sector**: Sector classification (e.g., HORTICULTURE, AGRICULTURE)
- **StateName**: State name (e.g., RAJASTHAN, MAHARASHTRA)
- **QueryText**: The actual question or query posed by the user
- **KccAns**: The answer or advisory provided

This metadata-rich format supports advanced filtering, region-specific analysis, and more detailed model training.

---

## Data Location

> **Download Link:**  
> [Click here to download the raw dataset](https://drive.google.com/file/d/1Tfe9d8uwPhxuVhfEkjgDeywKd_1dyhnc/view?usp=sharing)


---

## Instructions for Use

1. **Download the Dataset**  
   Download the dataset from the link above and save it to a suitable directory on your machine (e.g., `data/raw/`).

2. **Data Format**  
   - The dataset is currently provided as an zip file in csv format.  


3. **Do Not Use for Initial Setup**  
   This dataset is **not** required for the initial application setup. Complete the application installation and setup first using the provided instructions.

4. **Preprocessing for Application Training**  
   - Once you have the CSV/Excel file, you can preprocess the data (e.g., cleaning, normalization, feature selection) as needed for model training or analytics.
   - Save the preprocessed data in a separate directory (e.g., `data/processed/`).

5. **Further Steps**  
   - After preprocessing, you may use this data for fine-tuning the chatbot, building analytics dashboards, or conducting research.
   - Refer to the project’s data processing or model training documentation for next steps.

---
