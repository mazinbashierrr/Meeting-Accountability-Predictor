# Meeting Accountability Detector

Find the action items (tasks people commit to) in meeting transcripts, and
compare three models: a simple baseline (TF-IDF + Logistic Regression), RoBERTa,
and LLaMA-2.

This repo is the **data + baseline** part.

## What the notebook does

Open `meeting_accountability.ipynb` and run all cells. It:

1. Reads the AMI meeting files.
2. Labels each line: **1** if it's an action item, **0** if not.
3. Splits the data into train / dev / test.
4. Runs the baseline model and prints the scores.

It saves the labeled data to `utterances.parquet` and the splits to `splits/`
(as `.parquet` and `.jsonl`). The RoBERTa and LLaMA work should load the
`.jsonl` files so everyone uses the same rows.

## How to run

```bash
pip install -r requirements.txt
jupyter lab      # then open the notebook and run all cells
```

The data should be in `data/ami_public_manual_1.6.2/`. If it's not there yet:

```bash
mkdir -p data/ami_public_manual_1.6.2 && cd data/ami_public_manual_1.6.2
curl -L -O https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
unzip ami_public_manual_1.6.2.zip && rm ami_public_manual_1.6.2.zip
```

## How the labels work

AMI doesn't come with an "action item" label. But human annotators wrote a
summary for each meeting with an **Actions** section, and linked those action
sentences back to the lines that said them. So we mark a line as an action item
if it's linked to an Actions sentence. This gives **381 action items**, the same
number used in earlier research.

Section 3 of the notebook shows some action items next to their summary line, so
you can check the labels look right.

## What you should get

- 35 meetings, 115,388 lines, 381 action items (about 0.3%).
- Splits: 25 train / 5 dev / 5 test meetings.
- Baseline test score: action-item F1 around 0.38.

Action items are rare, so we don't use accuracy (a model that always says "no"
would score 99%). We use precision, recall, and F1 for the action-item class.
All three models should be scored this way.

## Data license

The AMI Meeting Corpus is free to use under CC BY 4.0. Please cite:

> J. Carletta et al. The AMI Meeting Corpus: A Pre-announcement. MLMI 2005.