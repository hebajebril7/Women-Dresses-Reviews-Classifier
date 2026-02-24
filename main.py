import joblib
import pandas as pd

def main():
    # 1) تحميل موديل SVM
    model = joblib.load("svm_model.pkl")  # حطي اسم ملف ال pkl تبعك

    # 2) تحميل CSV
    data = pd.read_csv("Women Dresses Reviews Dataset .csv")  # حطي اسم ملف البيانات تبعك

    # 3) اختيار البيانات
    data['title'] = data['title'].fillna("")
    data['review_text'] = data['review_text'].fillna("")

    X = (data['title'] + " " + data['review_text']).values

    # 4) توقع الأقسام
    clusters = model.predict(X)

    # 5) إضافة النتائج للـ CSV
    data["department_name"] = clusters

    # 6) طباعة النتائج
    print("Classification results:")
    print(data)

    # 7) حفظ النتائج في ملف جديد
    data.to_csv("output_with_predictions.csv", index=False)
    print("\nResults saved to output_with_predictions.csv")

if __name__ == "__main__":
    main()