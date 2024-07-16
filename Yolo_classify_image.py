import numpy as np
def yolo_classify_image(image_path, model, confidence_threshold):
    results = model(image_path)
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    prediction_dict = {names_dict[i]: probs[i] for i in range(len(probs))}
    sorted_predictions = sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True)
    max_prob_index = np.argmax(probs)
    predicted_category = names_dict[max_prob_index]
    certainty = probs[max_prob_index]

    if certainty > confidence_threshold:

        return predicted_category

    else:

        return "uncategorized"
