# Aura: Calibration-Free EMG Challenge — Overview & Data

**Competition:** [Aura: Calibration-Free EMG Challenge (PiEEG)](https://www.kaggle.com/competitions/vr-calibration-free-emg-challenge-pieeg/overview)  
**Data:** [Kaggle Data page](https://www.kaggle.com/competitions/vr-calibration-free-emg-challenge-pieeg/data)  
**Dataset version:** **2**, updated **29 October 2026**.  
**Source basis:** User-provided screenshots of the competition **Overview** and pasted **Dataset Description** (including its version-2 notes). This document records what those materials say; it does not fill in unspecified dataset details.

## 1. Objective and real-world use

Develop a **calibration-free gesture-recognition algorithm** for **Aura VR**, a wearable biosignal device developed by **PiEEG**, that recognizes hand gestures from muscle signals **across people and electrode placements**, including people whose recordings were **not used for training**.

- **Task:** Train an ML model to predict a hand gesture from sensor recordings.
- **Why it is challenging:** EMG activity varies between individuals. Patterns that work for one person might not generalize to a new wearer. The objective is to recognize **gestures, not participant identities**.
- **Data collection and labeling:** Performed using **Aura Dataset Lab**.
- **Intended application:** A person puts on Aura VR, makes a gesture, and the wearable interface recognizes it. The description mentions interactions with **computers, virtual environments, and assistive technologies**.
- **Modeling freedom:** Participants may build their own pipelines using filtering, normalization, temporal windowing, feature extraction, traditional ML, deep learning, or combinations. They may use **EMG alone, IMU alone, or both**; combining them may improve gesture discrimination and robustness across people.

## 2. Sensor modalities

| Modality | What the provided description states | Available test columns |
|---|---|---|
| **EMG** (electromyography) | Raw, time-varying signals representing electrical activity associated with muscle contractions | `ch1`–`ch8` |
| **IMU** (inertial measurement unit) | Motion measurements complementary to EMG | Accelerometer: `ax`, `ay`, `az`; gyroscope: `gx`, `gy`, `gz` |

- The **accelerometer** measures acceleration along three axes, including gravity's contribution.
- The **gyroscope** measures angular velocity around three axes.
- The test schema additionally lists `emg_new` and `imu_new`. The supplied description **does not define what these two fields mean**, so their interpretation should not be assumed.

## 3. Dataset files and split formats (v2)

All **data files are Excel (`.xlsx`)**; the **submission file is CSV (`.csv`)**.

| Files | Contents described by the organizers |
|---|---|
| `train/train_1.xlsx` … `train/train_28.xlsx` | **Full recording sessions with labels**, unchanged from dataset v1; all columns are present, including `label` and camera/hand-tracking columns |
| `test/test_1.xlsx` … `test/test_5.xlsx` | **Test actions without labels**, in the version-2 test format |
| `sample_submission.csv` | Example submission showing the identifiers and prediction format; follow its actual structure and ID order |

### Test data: exact version-2 format

- **One `action_id` is one gesture action** represented by a **1.4-second window**.
- Rows within each action are in **time order**.
- `t` **restarts at `0` for each action**.
- **Every signal channel is mean-centered per action**.
- **Listed test columns:**

  ```text
  action_id, t, emg_new, imu_new,
  ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8,
  ax, ay, az, gx, gy, gz
  ```

- **Not included in test:** `roll`, `pitch`, `yaw`, **all camera columns**, and **all hand-tracking columns**.

### Training data and train/test matching

- Training files contain **full sessions**, not the same already-windowed test representation.
- They include **all columns**, including `label` and camera/hand-tracking fields.
- **Only training rows with `label_source = guided` correspond to the test actions**; other rows are **transitions**.
- **Organizer tip:** To match the test format, cut the training data into **1.4-second windows** and **mean-center each signal channel separately within each window**.
- The description says the released training labels define the final set of prediction classes.

## 4. Prediction classes

Predict **one gesture per test action** using the **exact label spelling** found in training data:

| Label | Description |
|---|---|
| `rest` | Relax the hand without deliberately contracting the muscles |
| `fist` | Close the hand into a fist |
| `open` | Open the hand and spread the fingers |
| `point` | Extend the index finger while curling the other fingers |
| `pinch` | Touch the thumb to the index finger |
| `thumb-up` | Raise the thumb with the fingers loosely curled |

## 5. Evaluation and validation

- **Scoring metric:** **Macro-averaged F1** between predicted gesture labels and true labels.
- **Meaning:** F1 combines precision and recall for each class; **macro F1 is the unweighted average across classes**, so every gesture class has equal importance.
- **Range:** `0` to `1`; **higher is better**.
- **Cross-person evaluation:** The test set contains **participants excluded from the training set**. Generalization to previously unseen wearers is a central objective.
- **Dataset guidance for local validation:** When participant identifiers are available, split **by participant** so recordings from one person occur entirely in training **or** validation, never both.
- **Time-series integrity:** Preserve time order and recording boundaries when creating windows. Do **not** make a window spanning people or recording sessions.
- **Prevent target/test-feature leakage:** Gesture annotations and annotation-derived columns are targets or supporting information, **not sensor inputs**. Build the model from fields available at test time; in particular, don't depend on training-only camera/hand-tracking information.

## 6. Submission format

The **Overview's Submission File section** says:

- Submit **one predicted gesture per test `ID`**.
- Submit a **CSV with exactly two columns:** `ID` and `gesture`.
- Include **every test ID exactly once**, with **no missing IDs**, **no duplicates**, and **no extra index column**.
- Use the **exact label values** from training data.
- Follow **the structure and ID order** in `sample_submission.csv`.

The Overview shows this example:

```csv
ID,gesture
2,fist
5,open
6,pinch
```

**Important source discrepancy:** The **Dataset Description (v2)** refers to the sample submission as **`(action_id,gesture)`**, while the **Overview** specifies **`ID,gesture`**. It also names test actions `action_id`. These are different stated column names; **do not silently substitute one for the other**. Inspect the actual `sample_submission.csv` before generating the final submission and follow its exact header and ID order.

## 7. Final evaluation

The Dataset Description states that **top models will be verified live on the Aura device with new users after the competition ends**. No specific testing procedure, leaderboard-to-live-scoring formula, or date for live verification was provided in the supplied content.

## 8. What is not specified in the supplied Overview/Data material

The supplied text and screenshots do **not** establish the sample rate; signal units; exact training-sheet names or complete training-column schema; participant/session counts; class distributions; missing-value behavior; submission deadline or limit; dataset license; or the actual CSV header as stored in `sample_submission.csv`. Those details should be checked in the downloadable files or other competition pages rather than assumed.

---

**Source coverage:** Consolidated from the user-provided **Overview** screenshots (Overview, Description, Evaluation, Submission File) and **Dataset Description v2** text (files, sensor signals, target labels, preprocessing guidance, validation guidance, and final evaluation). No additional unstated rules or model requirements have been added.
