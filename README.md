# fakefail_report_tool


B1: Chọn daterange của các ngày tạo report với filter đã chọn sẳn https://metabase.ninjavan.co/question/25037-vn-lastmile-kpi-shopee?hub_region=HCM&hub_region=HN&hub_region=North&hub_region=South&dest_hub_date=2022-07-01~2022-11-03&only_shopee=no&aggregated=day&hub_breakdown=yes


B2: add link này vào directlink để đọc được folder các ngày. https://drive.google.com/drive/folders/1CrFk8raspvJK3FMDLMed4RRBgG1-WDIW?usp=share_link 

B3: Gọi funtion  read_pipeline(url_agg, str_time_from_, str_time_to_)

* url_agg = "" Link file volumns ở bước 1 đã tải "" 
* str_time_from_: thời điểm attempt_date bắt đầu
* str_time_to_: thời điểm attempt_date kết thúc
format: "mmmm-yy-dd"

Notice: QA khuyến khích custom code nhé!


# Cách chạy lấy data sample ff:
B1: Run query: https://metabase.ninjavan.co/question/62585-qa-vn-plan-for-ff-pod-manual-check-2022  --> link FF pod manual check


B2: Tải dữ liệu của query trên và up lên đây: https://drive.google.com/drive/folders/1WEDJfXnQg6q8Wmeriiw9dY6PpCJLYoNp  


B3: Import function sample_data từ file: data_sample.py
Câu lệnh import:
from data_sample import sample_data


B4: chạy câu lệnh sau để lưu dữ liệu:
sample_data(link_FF_pod_manual_check, start_day, end_day)

B5: Tải dữ liệu sample data là xong nè >_*

Ex:

![image](https://user-images.githubusercontent.com/74056907/209055774-428f6fba-5fa2-451e-bef1-4674bed8cc5a.png)

