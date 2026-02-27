import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ======================================
# HELPER FUNCTIONS (NO SKLEARN)
# ======================================

# Manual Train Test Split
def manual_train_test_split(X, y, test_size=0.2):
    np.random.seed(42)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split = int(len(X) * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]

    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


# ======================================
# GAUSSIAN NAIVE BAYES (MANUAL)
# ======================================

def gaussian_pdf(x, mean, var):
    eps = 1e-6
    coeff = 1.0 / np.sqrt(2 * np.pi * var + eps)
    exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))
    return coeff * exponent


def run_classification(df, target, features, test_size):

    X = df[features]
    y = df[target]

    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size)

    classes = np.unique(y_train)

    mean = {}
    var = {}
    prior = {}

    for c in classes:
        X_c = X_train[y_train == c]
        mean[c] = X_c.mean()
        var[c] = X_c.var()
        prior[c] = len(X_c) / len(X_train)

    predictions = []

    for i in range(len(X_test)):
        posteriors = []

        for c in classes:
            prior_prob = np.log(prior[c])
            likelihood = np.sum(
                np.log(gaussian_pdf(X_test.iloc[i], mean[c], var[c]))
            )
            posterior = prior_prob + likelihood
            posteriors.append(posterior)

        predictions.append(classes[np.argmax(posteriors)])

    predictions = np.array(predictions)

    # Accuracy
    accuracy = np.mean(predictions == y_test.values)

    # Confusion Matrix
    unique_classes = np.unique(y)
    cm = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)

    for true, pred in zip(y_test, predictions):
        i = list(unique_classes).index(true)
        j = list(unique_classes).index(pred)
        cm[i][j] += 1

    return accuracy, cm


# ======================================
# LINEAR REGRESSION (NORMAL EQUATION)
# ======================================

def run_regression(df, target, features, test_size):

    X = df[features]
    y = df[target]

    X = pd.get_dummies(X)

    # Convert target to numeric if needed
    if y.dtype == "object":
        y = pd.factorize(y)[0]

    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size)

    # Add bias column
    X_train = np.c_[np.ones(len(X_train)), X_train]
    X_test = np.c_[np.ones(len(X_test)), X_test]

    y_train = y_train.values
    y_test = y_test.values

    # Normal Equation
    theta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

    y_pred = X_test @ theta

    # MSE
    mse = np.mean((y_test - y_pred) ** 2)

    # R2
    ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
    ss_residual = np.sum((y_test - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)

    return mse, r2


# ======================================
# STREAMLIT UI
# ======================================

st.title("🧠 Machine Learning Dashboard (No Sklearn)")

uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    mode = st.radio(
        "🔎 Choose Section",
        ["📊 Exploratory Analytics", "🧠 Models"]
    )

    st.markdown("---")

    # ==============================
    # EXPLORATORY ANALYTICS
    # ==============================
    if mode == "📊 Exploratory Analytics":

        st.header("📊 Exploratory Data Analysis")

        st.dataframe(df.head())
        st.write("Shape:", df.shape)
        st.write("Missing Values:")
        st.write(df.isnull().sum())
        st.write("Data Types:")
        st.write(df.dtypes)

        st.subheader("Statistical Summary")
        st.dataframe(df.describe())

        numeric_df = df.select_dtypes(include=np.number)

        if not numeric_df.empty:
            fig, ax = plt.subplots()
            cax = ax.imshow(numeric_df.corr(), cmap="coolwarm")
            plt.colorbar(cax)
            st.pyplot(fig)
        else:
            st.info("No numeric columns found.")

    # ==============================
    # MODELS SECTION
    # ==============================
    else:

        problem_type = st.radio(
            "🔍 Select Problem Type",
            ["Classification", "Regression"]
        )

        target = st.selectbox("🎯 Select Target", df.columns)

        features = st.multiselect(
            "📊 Select Features",
            [col for col in df.columns if col != target]
        )

        test_size = st.slider("Train-Test Split", 0.1, 0.5, 0.2)

        if st.button("🚀 Evaluate Model"):

            if not features:
                st.error("Select at least one feature.")
            else:

                if problem_type == "Classification":
                    acc, cm = run_classification(
                        df, target, features, test_size
                    )

                    st.success(f"Accuracy: {acc:.4f}")
                    st.write("Confusion Matrix")
                    st.write(cm)

                else:
                    mse, r2 = run_regression(
                        df, target, features, test_size
                    )

                    st.success(f"MSE: {mse:.4f}")
                    st.success(f"R² Score: {r2:.4f}")

else:
    st.info("Upload a CSV file to begin.")