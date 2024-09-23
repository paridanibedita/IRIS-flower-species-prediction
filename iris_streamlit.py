import streamlit as st
import pandas as pd
import joblib

# Load the trained model from a saved file
def load_model():
    model = joblib.load('stacking_iris.pkl')
    return model

# Main function to run the app
def main():
    st.title("Iris Species Prediction")

    # Input form for new flower measurements
    st.subheader("Enter Iris Flower Measurements to Predict Species")
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.3)

    # Predict button
    if st.button("Predict Species"):
        # Create a new DataFrame with the user's input
        new_data = pd.DataFrame({
            'SepalLengthCm': [sepal_length],
            'SepalWidthCm': [sepal_width],
            'PetalLengthCm': [petal_length],
            'PetalWidthCm': [petal_width]
        })
        
        # Load the pre-trained model
        model = load_model()

        # Predict the species
        prediction = model.predict(new_data)
        
        # Decode the predicted species (0: Setosa, 1: Versicolor, 2: Virginica)
        species = ['Setosa', 'Versicolor', 'Virginica']
        predicted_species = species[prediction[0]]
        
        # Display the prediction
        st.write(f"### The predicted species is: **{predicted_species}**")

if __name__ == "__main__":
    main()
