import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


df = pd.read_excel("spam message detection model dataset.xlsx")
print(df.head())
print(df["Label"].value_counts())

X = df["Message"]
Y = df["Label"]
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    random_state=42
)


vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


y_train = Y_train.map({"ham": 0, "spam": 1})
y_test = Y_test.map({"ham": 0, "spam": 1})


model = MultinomialNB()
model.fit(X_train_vectorized, y_train)
print("The model training is completed")


y_pred = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {round(accuracy * 100, 2)}%")
print("Classification report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["ham", "spam"]
))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

while True:
    message = input("Enter your message: ")
    message_vectorized = vectorizer.transform([message])
    prediction = model.predict(message_vectorized)
    if prediction[0] == 1:
        print("Entered message is spam")
    else:
        print("Entered message is normal or ham message")
    exit_choice = int(input("Enter 1 to exit or 0 to continue: "))
    if exit_choice == 1:
        print("THANK YOU")
        break

print("This is only a model. It may not predict correctly every time.")
