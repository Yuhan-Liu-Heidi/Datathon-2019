import pandas as pd


def load(file):

    import ast
    df = pd.read_csv(file)

    # convert the column values from literal string to dictionary
    df['ltiFeatures'] = df['ltiFeatures'].apply(ast.literal_eval)
    df['stiFeatures'] = df['stiFeatures'].apply(ast.literal_eval)

    return df


def main():
    # load all data
    training = load("training.csv")
    validation = load("validation.csv")
    interest_topics = pd.read_csv("interest_topics.csv")

    # inspect data
    interest_topics.head()

    training.head()

    validation.head()


if __name__ == "__main__":
    main()
