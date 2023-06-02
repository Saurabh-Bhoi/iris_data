from flask import Flask, jsonify, render_template, request

from project_app.utils import IrisData

app = Flask(__name__)

@app.route("/")
def hello_flask():
    print("Welcome to Iris Prediction System")
    return render_template("index.html")

@app.route("/predicted_data", methods = ["GET","POST"])
def get_predicted_data():
    if request.method == "GET":
        print("We are in get method")

        SepalLengthCm = float(request.args.get("SepalLengthCm"))
        SepalWidthCm = float(request.args.get("SepalLengthCm"))
        PetalLengthCm = float(request.args.get("SepalLengthCm"))
        PetalWidthCm = float(request.args.get("SepalLengthCm"))

        species = IrisData(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
        iris_data = species.get_iris_prediction()
        
        return render_template("index.html", prediction = iris_data)
    
print("__name__-->",__name__)

if __name__ == "__main__":
    app.run(host= "0.0.0.0", port= 5005, debug = True)
