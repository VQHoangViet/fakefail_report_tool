# fakefail_report_tool


B1: Chọn daterange của các ngày tạo report với filter đã chọn sẳn https://metabase.ninjavan.co/question/25037-vn-lastmile-kpi-shopee?hub_region=HCM&hub_region=HN&hub_region=North&hub_region=South&dest_hub_date=2022-07-01~2022-11-03&only_shopee=no&aggregated=day&hub_breakdown=yes


B2: add link này vào directlink để đọc được folder các ngày. https://drive.google.com/drive/folders/1CrFk8raspvJK3FMDLMed4RRBgG1-WDIW?usp=share_link 

B3: Gọi funtion  read_pipeline(url_agg, str_time_from_, str_time_to_)

* url_agg = "" Link file volumns ở bước 1 đã tải "" 
* str_time_from_: thời điểm attempt_date bắt đầu
* str_time_to_: thời điểm attempt_date kết thúc
format: "mmmm-yy-dd"

Notice: QA khuyến khích custom code nhé!