from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

predictionEndpoint = "YOUR_PREDICTION_ENDPOINT"
predictionKey = "YOUR_PREDICTION_KEY"
projectId = "YOUR_PROJECT_ID"
modelName = "YOUR_MODEL_NAME"


def main():
        #authentication 
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": predictionKey})
        prediction_client = CustomVisionPredictionClient(endpoint = predictionEndpoint, credentials = credentials)

        image_file = r"images/test.jpg"

        #detcting objects
        with open(image_file, mode = 'rb') as image_data:
            results = prediction_client.classify_image(projectId, modelName, image_data)
        print(results)
        for prediction in results.predictions:
            print("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))

if __name__ == "__main__":
    main() 
