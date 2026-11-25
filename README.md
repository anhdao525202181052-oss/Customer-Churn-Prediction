# Dự án Dự đoán Khách hàng Rời bỏ (Customer Churn Prediction)

## 1. Tổng quan đề tài

Đề tài tập trung vào **dự đoán khả năng khách hàng rời bỏ trong thương mại điện tử** dựa trên dữ liệu hành vi và lịch sử giao dịch của khách hàng.

Mục tiêu chính của đề tài là xác định những khách hàng có nguy cơ rời bỏ, phân tích đặc điểm hành vi của khách hàng và tìm hiểu những yếu tố có ảnh hưởng đến khả năng rời bỏ.

Đề tài kết hợp các phương pháp phân tích dữ liệu, phân khúc khách hàng, học máy và giải thích mô hình nhằm hỗ trợ doanh nghiệp đưa ra các chiến lược giữ chân khách hàng phù hợp.

---

## 2. Bộ dữ liệu

Đề tài sử dụng bộ dữ liệu **Brazilian E-Commerce Public Dataset by Olist**.

Đây là bộ dữ liệu thương mại điện tử tại Brazil, bao gồm thông tin về khách hàng, đơn hàng, sản phẩm, thanh toán, người bán, đánh giá của khách hàng và quá trình giao hàng.

Dữ liệu ban đầu được lưu trong nhiều bảng khác nhau. Trong quá trình thực hiện, các bảng dữ liệu được làm sạch, xử lý và kết hợp nhằm xây dựng bộ dữ liệu ở cấp độ khách hàng phục vụ cho việc phân tích và xây dựng mô hình dự đoán khách hàng rời bỏ.

---

## 3. Quy trình thực hiện

Đề tài được thực hiện theo các bước chính sau:

1. Thu thập và tìm hiểu dữ liệu
2. Làm sạch và tiền xử lý dữ liệu
3. Phân tích khám phá dữ liệu (EDA)
4. Xây dựng các đặc trưng phục vụ phân tích
5. Phân tích RFM
6. Phân khúc khách hàng bằng K-Means
7. Xác định khách hàng rời bỏ (Churn)
8. Lựa chọn đặc trưng
9. Xây dựng mô hình học máy
10. Đánh giá và so sánh mô hình
11. Giải thích mô hình bằng SHAP
12. Đề xuất ứng dụng trong kinh doanh

---

## 4. Phân tích RFM

Phương pháp **RFM** được sử dụng để phân tích hành vi mua hàng của khách hàng dựa trên ba yếu tố:

- **Recency (R):** Khoảng thời gian kể từ lần mua hàng gần nhất của khách hàng.
- **Frequency (F):** Tần suất mua hàng của khách hàng.
- **Monetary (M):** Tổng giá trị chi tiêu của khách hàng.

Các chỉ số RFM giúp mô tả hành vi của từng khách hàng và được sử dụng trong quá trình phân tích cũng như phân khúc khách hàng.

---

## 5. Phân khúc khách hàng bằng K-Means

Thuật toán **K-Means Clustering** được sử dụng để phân nhóm những khách hàng có đặc điểm hành vi tương đồng.

Việc phân khúc giúp nhận diện sự khác biệt giữa các nhóm khách hàng, từ đó hiểu rõ hơn về hành vi mua sắm và đặc điểm của từng nhóm trước khi xây dựng mô hình dự đoán khách hàng rời bỏ.

---

## 6. Dự đoán khách hàng rời bỏ

Sau quá trình tiền xử lý dữ liệu và xây dựng đặc trưng, bài toán dự đoán khách hàng rời bỏ được xây dựng dưới dạng bài toán phân loại trong học máy.

Đề tài sử dụng và so sánh hai mô hình chính:

### Logistic Regression

Logistic Regression được sử dụng làm mô hình cơ sở để dự đoán xác suất khách hàng rời bỏ. Đây là mô hình tương đối đơn giản và có khả năng diễn giải tốt.

### XGBoost

XGBoost được sử dụng nhằm khai thác các mối quan hệ phức tạp và phi tuyến giữa các đặc điểm của khách hàng và khả năng rời bỏ.

Kết quả của các mô hình được đánh giá và so sánh nhằm lựa chọn phương pháp phù hợp hơn với dữ liệu của đề tài.

---

## 7. Đánh giá mô hình

Các mô hình được đánh giá thông qua những chỉ số phổ biến trong bài toán phân loại, bao gồm:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Việc đánh giá dựa trên nhiều chỉ số giúp phản ánh đầy đủ hơn hiệu quả của mô hình thay vì chỉ dựa vào độ chính xác tổng thể.

---

## 8. Giải thích mô hình bằng SHAP

Đề tài sử dụng **SHAP (SHapley Additive exPlanations)** để giải thích kết quả dự đoán của mô hình học máy.

SHAP giúp xác định những đặc trưng có ảnh hưởng nhiều nhất đến khả năng một khách hàng được dự đoán là rời bỏ hoặc tiếp tục mua hàng.

Thông qua đó, kết quả của mô hình không chỉ dừng lại ở việc dự đoán mà còn giúp làm rõ các yếu tố liên quan đến hành vi rời bỏ của khách hàng.

---

## 9. Ý nghĩa và ứng dụng trong kinh doanh

Kết quả của đề tài có thể hỗ trợ doanh nghiệp trong việc:

- Nhận diện nhóm khách hàng có nguy cơ rời bỏ.
- Hiểu rõ hơn về hành vi mua sắm của khách hàng.
- Phân khúc khách hàng dựa trên đặc điểm hành vi.
- Xác định các yếu tố có liên quan đến khả năng rời bỏ.
- Hỗ trợ xây dựng các chương trình giữ chân khách hàng.
- Hỗ trợ hoạt động marketing và quản trị quan hệ khách hàng.

Thay vì áp dụng cùng một chính sách cho tất cả khách hàng, doanh nghiệp có thể dựa trên hành vi và mức độ rủi ro của từng nhóm để ưu tiên các hoạt động chăm sóc và giữ chân phù hợp hơn.

---

## 10. Mã nguồn

Toàn bộ quá trình xử lý dữ liệu, phân tích, xây dựng mô hình, đánh giá và trực quan hóa kết quả được thực hiện trong Jupyter Notebook của dự án.

**File mã nguồn:** `Customer_Churn_Prediction.ipynb`

Notebook có thể được mở và chạy bằng **Google Colab** hoặc các môi trường hỗ trợ Jupyter Notebook.

---

## 11. Công cụ và thư viện sử dụng

Đề tài được thực hiện bằng **Python** với các thư viện phục vụ xử lý dữ liệu, trực quan hóa và học máy, bao gồm:

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP

**Môi trường và công cụ sử dụng:**

- Google Colab
- Jupyter Notebook
- GitHub

---

## 12. Mục đích thực hiện

Đây là đề tài học thuật thuộc chương trình **Hệ thống Thông tin Quản lý tại UEH**.

Thông qua bộ dữ liệu thương mại điện tử của Olist, đề tài trình bày quy trình từ xử lý và khám phá dữ liệu, phân tích hành vi khách hàng, phân khúc khách hàng đến xây dựng và giải thích mô hình dự đoán khách hàng rời bỏ.

Kết quả nghiên cứu cho thấy cách các phương pháp phân tích dữ liệu và học máy có thể được áp dụng vào dữ liệu giao dịch thực tế nhằm hỗ trợ doanh nghiệp hiểu khách hàng và đưa ra các quyết định kinh doanh dựa trên dữ liệu.
