import pickle


def main():

    # Load the results from the pickle file
    with open("validation_prediction.pkl", "rb") as f:
        validation_predictions = pickle.load(f)
    print(validation_predictions)
    output_labels = []
    for message in validation_predictions:
        # print(message)
        # break
        lst = message.strip().split("\n")
        print(len(lst))
        # break
        # for predict in lst:
        #     output_labels.append(
        #         int(predict.strip().split('Classification: ')[1]))
        # print(output_labels)
        # break
    # print(len(output_labels))


if __name__ == '__main__':
    main()
