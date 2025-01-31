from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        
        if file.filename == "":
            return redirect(request.url)
        
        if file and file.filename.endswith(".csv"):
            df = pd.read_csv(file)
            churn_rate = calculate_churn_rate(df)
            return render_template("result.html", churn_rate=churn_rate)
    
    return render_template("index.html")

@app.route("/manual", methods=["GET", "POST"])
def manual():
    if request.method == "POST":
        total_customers = int(request.form["total_customers"])
        churned_customers = int(request.form["churned_customers"])

        if total_customers == 0:
            churn_rate = "Error: Total customers cannot be zero."
        else:
            churn_rate = (churned_customers / total_customers) * 100
            churn_rate = round(churn_rate, 2)
        
        return render_template("result.html", churn_rate=churn_rate)
    
    return render_template("manual.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        if username == "admin" and password == "password":
            return redirect(url_for("home"))
        else:
            return "Invalid credentials. Please try again."
    
    return render_template("login.html")

def calculate_churn_rate(df):
    if "Churn" not in df.columns:
        return "Error: 'Churn' column not found in the dataset."
    
    total_customers = len(df)
    churned_customers = df[df["Churn"] == 1].shape[0]
    
    if total_customers == 0:
        return "Error: No data found in the dataset."
    
    churn_rate = (churned_customers / total_customers) * 100
    return round(churn_rate, 2)

if __name__ == "__main__":
    app.run(debug=True)
