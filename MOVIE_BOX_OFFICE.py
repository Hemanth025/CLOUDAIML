
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("Movie_regression.csv")

# Handling missing values
data["Time_taken"].fillna(value=data["Time_taken"].mean(), inplace=True)

# Scale Budget and Collection
data["Budget"] = data["Budget"] / 100
data["Collection"] = data["Collection"] / 100

# Visualizing key relationships
sns.scatterplot(x="Lead_ Actor_Rating", y="Collection", data=data)
plt.title("Collection vs Actor Rating")
plt.show()

sns.scatterplot(x="Director_rating", y="Collection", data=data)
plt.title("Collection vs Director Rating")
plt.show()

# One-hot encoding for categorical variables
data = pd.get_dummies(data, columns=["3D_available", "Genre"])
data.drop(["Multiplex coverage", "Time_taken", "Num_multiplex", "3D_available_NO"], axis=1, inplace=True)

# Defining feature and target variables
x = data.loc[:, data.columns != "Collection"]
y = data["Collection"]

# Splitting data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model training with Random Forest
rf_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=0)
rf_model.fit(x_train_scaled, y_train)

# Model prediction and evaluation
y_pred = rf_model.predict(x_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-Squared: {r2}")

# Predict movies above 1000 crores
def prediction_calculator(n):
    for i in range(n):
        print(f"\nENTER THE INPUTS FOR MOVIE {i + 1}:")
        ME = float(input("Marketing expense (in crores): ")) / 100
        PE = float(input("Production expense (in crores): ")) / 100
        Budget = float(input("Budget (in crores): ")) / 100
        ML = float(input("Movie Length (in mins): "))
        Act_rate = float(input("Lead Actor rating (1-10 IMDb scale): "))
        Actr_rate = float(input("Lead Actress rating (1-10 IMDb scale): "))
        DR = float(input("Director rating (1-10 IMDb scale): "))
        PR = float(input("Producer rating (1-10 IMDb scale): "))
        CR = float(input("Critic rating (1-10 scale): "))
        TV = float(input("Trailer views (in lakhs): "))
        TH = float(input("Twitter hashtags: "))
        Avg_act = float(input("Average actors: "))
        three_d = int(input("3D (1-yes / 0-no): "))
        
        genre = input("Genre (action/comedy/drama/thriller): ").lower()
        ga, gc, gd, gt = 0, 0, 0, 0
        if genre == "action":
            ga = 1
        elif genre == "comedy":
            gc = 1
        elif genre == "drama":
            gd = 1
        elif genre == "thriller":
            gt = 1
        
        # Prepare input array
        c = np.array([ME, PE, Budget, ML, Act_rate, Actr_rate, DR, PR, CR, TV, TH, Avg_act, three_d, ga, gc, gd, gt])
        c_rs = c.reshape(1, -1)
        
        # Predict with scaled input
        pred_scaled = scaler.transform(c_rs)
        pred = rf_model.predict(pred_scaled)[0] * 100 / 15
        
        # Display predictions
        if pred > 1000:
            print(f"\n🎉 The predicted collection is {pred:.2f} crores (Above 1000 crores!) 🎬")
        else:
            print(f"\nThe predicted collection is {pred:.2f} crores. Aim higher next time!")

# Get user input for number of use cases
use_case = int(input("\nENTER NUMBER OF MOVIES TO PREDICT: "))
prediction_calculator(use_case)
