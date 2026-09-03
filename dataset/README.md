# Grocery Store Dataset

## Dataset Name

**Grocery Store Dataset**

## Purpose

This dataset is used for the **Smart Supermarket Product Identification** project.

The goal is to train a computer vision model that can identify grocery products from images captured in supermarket environments.

## Dataset Source

The dataset is the **Grocery Store Dataset** by Marcus Klasson.

Official repository:

https://github.com/marcusklasson/GroceryStoreDataset

## Dataset Statistics

| Property                  | Value |
| ------------------------- | ----: |
| Natural images            | 5,421 |
| Fine-grained classes      |    81 |
| Coarse-grained categories |    43 |
| Training images           | 2,640 |
| Validation images         |   296 |
| Test images               | 2,485 |

The dataset contains natural product images captured in grocery store environments.

## Class Structure

The dataset provides two levels of classification:

### Fine-grained classes

There are **81 fine-grained product classes**.

Examples:

* Golden-Delicious
* Granny-Smith
* Pink-Lady
* Royal-Gala
* Arla-Standard-Milk
* Tropicana-Apple-Juice

The fine-grained classes will be used as the primary target for product identification.

### Coarse-grained categories

The dataset also provides broader product categories.

Examples:

* Apple
* Milk
* Juice
* Yoghurt
* Tomato
* Pepper

The relationship between fine-grained classes and coarse-grained categories is defined in `classes.csv`.

## Dataset Splits

The dataset provides three splits:

```text
train/
val/
test/
```

### Training Set

* 2,640 images
* All 81 fine-grained classes are represented

### Validation Set

* 296 images
* 60 of the 81 fine-grained classes are represented

### Test Set

* 2,485 images
* All 81 fine-grained classes are represented

The validation split will be considered carefully during evaluation because not all 81 classes have validation samples.

## Dataset Files

Important dataset files include:

```text
classes.csv
train.txt
val.txt
test.txt
train/
val/
test/
```

`classes.csv` contains the mapping between fine-grained and coarse-grained product categories.

The TXT files contain the image paths and corresponding class labels.

## Image Information

Most images are approximately:

```text
348 × 348 pixels
```

Some images have portrait or landscape dimensions.

The images are primarily JPEG files.

## Data Quality

During initial dataset inspection:

* No corrupted JPEG images were detected.
* No duplicate natural images were detected across the train, validation, and test splits.
* Image paths and class labels were found to be consistent with the dataset structure.

## Class Imbalance

The dataset is not perfectly balanced.

The number of training images per fine-grained class ranges approximately from:

```text
16 → 70 images
```

Therefore, model evaluation will not rely only on overall accuracy.

Additional metrics such as:

* Precision
* Recall
* Macro F1-score
* Weighted F1-score
* Confusion Matrix

will be considered.

## Dataset Usage

The actual dataset images should **not be committed to the Git repository**.

The repository will contain the dataset documentation and the code required to work with the dataset.

The actual dataset should be downloaded separately and stored locally during development.

## Planned Machine Learning Task

The primary task will be:

```text
Input Image
     ↓
Image Preprocessing
     ↓
Deep Learning Model
     ↓
81-Class Product Classification
     ↓
Predicted Product
     ↓
Coarse Category
```

Example:

```text
Input:
Image of Granny Smith apple

Prediction:
Fine-grained class: Granny-Smith
Coarse category: Apple
```

## Project Note

This dataset was selected because it contains natural grocery-store images and provides fine-grained product classes suitable for developing a supermarket product identification system.
