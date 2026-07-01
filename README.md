# JCDSBDGAM-11_Remedial


## **Business Understanding**

#### **Latar Belakang**

Operasional bisnis hotel sangat bergantung pada tingkat keterisian kamar (Occupancy Rate) untuk memaksimalkan pendapatan (Revenue). Masalah terbesar yang sering dihadapi oleh manajemen hotel adalah **tingginya angka pembatalan pemesanan kamar (Booking Cancellation)** mendadak oleh tamu. Ketika seorang tamu membatalkan pesanan di menit-menit akhir, hotel akan kehilangan potensi pendapatan dari kamar tersebut, karena sangat sulit menjual kembali kamar kosong dalam waktu yang singkat.

Berdasarkan analisis pada data, tingkat pembatalan kamar (Cancellation Rate) di hotel ini sangat mengkhawatirkan, yaitu menyentuh angka 28% (artinya, dari setiap 100 pesanan yang masuk, ada 28 pesanan yang berakhir batal). Tren ini didominasi oleh segmen Agen Perjalanan Online (Online TA) dan pelanggan individu (Transient). Angka 28% ini merupakan sebuah kebocoran finansial yang sangat besar bagi manajemen jika dibiarkan tanpa tindakan mitigasi.

Untuk melawan kerugian akibat pembatalan 28% tersebut, hotel tidak boleh pasif dan membiarkan kamar kosong begitu saja. Hotel wajib menerapkan strategi "Overbooking", yaitu menerima jumlah pesanan kamar melebihi kapasitas fisik asli yang dimiliki hotel (misalnya menerima 105 pesanan untuk kapasitas 100 kamar) tanpa secara signifikan meningkatkan revenue loss. Tujuannya untuk memastikan bahwa ketika hari H tiba dan ada tamu yang batal datang, kamar tersebut sudah ada penggantinya secara otomatis, sehingga tingkat hunian tetap mencapai 100% (Full Occupancy).

---

#### **Permasalahan Utama**
- Apa faktor - faktor yang mempengarugi tingginya nilai cancelled rate?
- Bagaimana caranya perusahaan bisa memprediksi pengunjung yang memiliki resiko tinggi untuk membatalkan reservasi menggunakan model pembelajaran mesin berbasis data dan menentukan tindakan strategis yang tepat untuk mencegah kehilangan pelanggan?

---

#### **Goals**
- Untuk mengidentifikasi faktor-faktor yang berkontribusi terhadap nilai cancelled rate yang tinggi. 
- Untuk mengembangkan model pembelajaran mesin berbasis data untuk memprediksi pelanggan berisiko tinggi yang akan membatalkan reservasi dan memberikan rekomendasi strategis untuk mencegah kehilangan pelanggan dan meningkatkan pengambilan keputusan bisnis.

---

#### **Stakeholders** 
Model ini dirancang khusus untuk digunakan oleh **tim Revenue Management** sebagai sistem pendukung keputusan otomatis saat mengimplementasikan kuota harian overbooking.

---

#### **Pendekatan Analisis**

1. **Pengumpulan dan Praproses Data**
- Menggunakan data pelanggan historis.
- Melakukan pembersihan data dan transformasi variabel kategorikal.
2. **Exploratory Data Analysis (EDA)**
- Mengidentifikasi faktor yang mempengaruhi terhadap cancelled.
- Melakukan analisis statistik untuk memeriksa hubungan antara fitur dan is_cancelled.
3. **Modelling & Evaluation**
- Menggunakan algoritma klasifikasi untuk memprediksi pengunjung yang akan membatalkan reservasinya.
- Memilih metrik evaluasi yang tepat.
4. **Implementation & Business Action**
- Mengidentifikasi pengunjung berisiko tinggi yang akan membatalkan reservasinya berdasarkan prediksi hasil.
- Tandai akun tamu yang berisiko tinggi akan membatalkan reservasi di sistem PMS.

---

#### **Confussion Matrix**

|  | Prediksi tidak dibatalkan **(0)** | Prediksi dibatalkan **(1)**|
|-------------------|-----------------|-------------------|
| **Aktual tidak Dibatalkan (0)** |**True Negative (TN)**<br> Model memprediksi bahwa pengunjung akan datang padahal sebenarnya akan datang |**False Positive (FP)**<br> Model memprediksi bahwa pengunjung akan batal padahal sebenarnya akan datang |
| **Aktual dibatalkan (1)** |**False Negative (FN)**<br> Model memprediksi bahwa pengunjung akan datang padahal sebenarnya dia akan batal datang |**True Positive (TP)**<br> Model memprediksi bahwa pengunjung akan batal padahal sebenarnya dia akan batal datang|

Strategi overbooking adalah pedang bermata dua. Tantangan terbesarnya adalah ketidakpastian menebak siapa tamu yang benar-benar akan batal.
- **Jika Salah Prediksi (False Positive / FP)**: Model memprediksi tamu A akan **Batal**, sehingga hotel menjual kamarnya ke tamu B. Namun pada hari-H, tamu A ternyata **Datang**. Kondisi ini menyebabkan hotel kekurangan kamar (Overbooked Crisis). Hotel terpaksa menanggung denda kompensasi yang sangat mahal, yaitu memindahkan tamu ke hotel kompetitor setara dengan biaya ganti rugi sebesar **300 Euro** (FP Cost) per tamu.
- **Jika Salah Prediksi (False Negative / FN):** Model memprediksi tamu akan **Datang**, tetapi ternyata mereka **Batal**. Kamar menjadi kosong dan hotel menderita kerugian potensi penjualan kamar (Opportunity Loss) sebesar **105 Euro** (FN Cost).


---

#### **Evaluasi Metrics**

Karena biaya ganti rugi akibat salah prediksi tamu (FP = 300 Euro) jauh lebih mahal (hampir 3 kali lipat) dibandingkan membiarkan kamar kosong (FN = 105 Euro), maka False Positive (FP) adalah musuh utama bisnis hotel ini.

Untuk menyelesaikan masalah ini, solusi data analisis yang diterapkan adalah membangun model Machine Learning dengan mengoptimasi metrik $F_{0.5}\text{-Score}$  

Metrik $F_{0.5}$ dipilih karena secara matematis memberikan bobot hukuman yang lebih berat pada kesalahan False Positive (Precision-focused) tanpa secara signifikan meningkatkan false negatif. Model dipaksa untuk tidak sembarangan menebak orang akan batal, kecuali jika model benar-benar yakin 100%.

**Formula Metrics**

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

$$F_\beta = \frac{1 + \beta^2}{\frac{\beta^2}{\text{Recall}} + \frac{1}{\text{Precision}}}$$

untuk **F-0.5 Score, β = 0.5**

Keterangan:
- **Presisi** = TP / (TP + FP) - Proporsi pengunjung yang diprediksi akan membatalkan pemesanan dan benar-benar membatalkan pemesanan

- **Recall** = TP / (TP + FN) - Proporsi pelanggan yang benar-benar membatalkan pemesanan dan teridentifikasi dengan tepat

- **β = 0.5** berarti precision diberi bobot **2x** lebih penting daripada recall
---
#### **Referensi & Acuan Penentuan Nilai Cost:**

**1. Median ADR = 99 euro/malam**

**2. False Negative (FN)** terjadi saat model menebak seorang tamu akan datang, tapi ternyata mereka batal. Akibatnya, kamar menjadi kosong dan hotel menderita kerugian sebesar **€105**, yang dihitung dari:

- **Kehilangan Pendapatan Kamar (Lost Revenue): €94**, Kamar batal di menit-menit akhir (last minute). Berdasarkan data industri, peluang hotel bisa menjual kembali kamar kosong tersebut secara mendadak sangat kecil, yaitu hanya 5%. Artinya, hotel murni kehilangan 95% potensi uang dari harga kamar asli (€99 $(\times )$ 0.95 = €94).

- **Biaya Operasional yang Terbuang: €10**, Meskipun kamar kosong dan tidak ada orangnya, hotel tetap mengeluarkan biaya listrik, AC, dan upah staf housekeeping yang sudah telanjur membersihkan kamar tersebut. Berdasarkan laporan riset dunia (STR Global Hotel Report), biaya operasional ini memakan sekitar 10% dari harga kamar (sekitar €10).

`Total FN Cost: €94 + €10 = €104 ≈ €105`

**3. False Positive (FP)** terjadi saat model menebak seorang tamu akan batal, sehingga hotel menjual kamarnya ke orang lain (overbooking). Namun pada hari H, tamu asli tersebut ternyata datang. Hotel mengalami krisis kekurangan kamar dan harus menanggung denda total sebesar **€300**, karena tiga komponen ini:

- **Biaya Relokasi Tamu (Walking Cost): €150**, Karena kamar di hotel kita habis, kita wajib memesankan dan membayar kamar di hotel kompetitor yang setara atau lebih mewah untuk tamu tersebut, ditambah biaya taksi/transportasi untuk mengantar mereka ke hotel baru tersebut. Berdasarkan riset ilmiah Cornell Hospitality, biaya rata-rata pengalihan tamu ini berkisar antara €130 - €180. satu malam menginap di hotel setara dengan €99 - €150 dan Transport ke hotel lain sekitar €15 - €30, Kita ambil jalan tengah di angka €150.

- **Biaya Kompensasi dan Permintaan Maaf: €75**, Tamu pasti akan marah karena kamarnya diberikan ke orang lain. Untuk meredam amarah dan menjaga nama baik, hotel wajib memberikan uang kompensasi tunai serta voucher makan gratis di restoran. Menurut panduan American Hotel & Lodging Association (AHLA), biaya santunan ini memakan biaya sekitar €75.

- **Kerugian Nama Baik / Ulasan Negatif (Reputation Damage): €75**, Tamu yang kecewa kemungkinan besar akan menulis ulasan bintang 1 di TripAdvisor atau Booking.com. Berdasarkan jurnal ilmiah internasional (Ye, Law, & Gu, 2009), 1 ulasan negatif di internet bisa membuat hotel kehilangan potensi pendapatan masa depan sebesar €100 hingga €200 karena calon tamu lain menjadi takut untuk memesan. Risiko kerugian nama baik per kejadian ini kita taksir minimal di angka €75.

`Total FP Cost: Walking (€150) + Kompensasi (€75) + Reputasi (€75) = €300`

**Referensi:**
- STR Global Hotel Report (2017) "Fixed operating cost per room: 8-12% of ADR", www.str.com
- Cornell Hospitality Report (2015) "The Cost of Overbooking", www.chr.cornell.edu
- American Hotel & Lodging Association (AHLA) "Best Practices in Overbooking Management", www.ahla.com
- Ye, Q., Law, R., & Gu, B. (2009) "The Impact of Online Reviews on Hotel Revenue", International Journal of Hospitality Management, DOI: 10.1016/j.ijhm.2008.09.011

---
## **Data Understanding**
**1. Waktu pemesanan & menginap**
| Nama Fitur | Keterangan |
|-------------------|-----------------|
`lead_time`|Jumlah hari dari masukkan pemesanan ke dalam PMS sampai tanggal kedatangan |
|`arrival_date_year` |Tahun tanggal kedatangan|
|`arrival_date_month` |Bulan tanggal kedatangan|
|`arrival_date_week_number` |Nomor minggu pertahun untuk tanggal kedatangan|
|`arrival_date_day_of_month` |Hari tanggal kedatangan|
|`stays_in_weekend_nights` |Jumlah malam akhir pekan (Sabtu atau Minggu) tamu menginap di hotel|
|`stays_in_week_nights` |Jumlah malam hari kerja (Senin sampai Jumat) tamu menginap di hotel|

**2. profil tamu**
| Nama Fitur | Keterangan |
|-------------------|-----------------|
|`adults` |Jumlah orang dewasa|
|`children` |Jumlah anak-anak|
|`babies` |Jumlah bayi|
|`country` |Negara asal tamu, Kategori-kategori tersebut direpresentasikan dalam format ISO 3155–3:2013.|
|`agent`|Identitas agen perjalanan yang melakukan pemesanan|
|`company`|Identitas perusahaan yang melakukan pemesanan atau bertanggung jawab atas pembayaran pemesanan.|
|`hotel`|Tipe Hotel (H1 = Resort Hotel/ H2 = City Hotel) |
| `market_segment`|Tipe segmen pasar (TA = Agen Perjalanan/ TO = Operator Tur)|
|`distribution_channel` | Saluran distribusi pemesanan (TA = Agen Perjalanan/ TO = Operator Tur/ Direct = berarti langsung)|

**3. Detail kamar & pemesanan**
| Nama Fitur | Keterangan |
|-------------------|-----------------|
|`meal` |Jenis makanan yang dipesan (SC = tanpa paket makan/ BB = Sarapan dan Penginapan/ HB = Sarapan dan makan malam/ FB = sarapan, makan siang, dan makan malam)|
| `reserved_room_type` |Kode tipe kamar yang dipesan. |
|`assigned_room_type` |Kode tipe kamar yang diberikan saat datang |
|`customer_type` | Tipe pengunjung (Contract = ketika pemesanan memiliki alokasi atau jenis kontrak lain yang terkait dengannya/ Grup = ketika pemesanan terkait dengan sebuah grup/ Transient = ketika pemesanan bukan bagian dari grup atau kontrak, dan tidak terkait dengan pemesanan tamu tunggal lainnya/ Transient-party = ketika pemesanan bersifat tamu tunggal, tetapi terkait dengan setidaknya satu pemesanan tamu tunggal lainnya)|
| `is_repeated_guest`|Nilai yang menunjukkan apakah nama pemesanan berasal dari tamu berulang (1) atau bukan (0) |
|`previous_cancellations` | Jumlah pemesanan sebelumnya yang dibatalkan oleh pelanggan sebelum pemesanan saat ini.|
|`previous_bookings_not_canceled` | Jumlah pemesanan sebelumnya yang tidak dibatalkan oleh pelanggan sebelum pemesanan saat ini.|
| `required_car_parking_spaces`| Jumlah tempat parkir mobil yang dibutuhkan oleh pelanggan|
|`total_of_special_requests` | Jumlah permintaan khusus yang diajukan oleh pelanggan (misalnya tempat tidur kembar atau lantai atas)|
| `days_in_waiting_list`|Jumlah hari pemesanan berada dalam daftar tunggu sebelum dikonfirmasi kepada pelanggan |
| `booking_changes`| Jumlah perubahan/amandemen yang dilakukan pada pemesanan sejak pemesanan dimasukkan ke dalam PMS hingga saat check-in atau pembatalan|
|`reservation_status` |Status reservasi terakhir (Canceled = pemesanan dibatalkan oleh pelanggan/ Check-Out = pelanggan telah check-in tetapi sudah check-out/ No-Show = pelanggan tidak check-in dan tidak memberi tahu hotel alasannya)|
|`reservation_status_date` |Tanggal saat status terakhir ditetapkan |
|`is_canceled` (Fitur Target) |Nilai yang menunjukkan apakah pemesanan dibatalkan (1) atau tidak (0) |

**4. Risiko & keuangan**
| Nama Fitur | Keterangan |
|-------------------|-----------------|
|`deposit_type` |Indikasi apakah pelanggan melakukan pembayaran uang muka untuk menjamin pemesanan (No Deposit = tidak ada uang muka yang dibayarkan/ Non Refund = uang muka dibayarkan senilai total biaya menginap/ Refundable = uang muka dibayarkan dengan nilai di bawah total biaya menginap)|
|`adr` |Tarif Harian Rata-Rata (membagi jumlah total transaksi penginapan dengan total jumlah malam menginap)|

></div>
****
## **3. Business Insight**

**Ketergantungan Segmen TA/Online**

Pendapatan hotel sangat bergantung pada **Agen Perjalanan Online (OTA)** dan pelanggan individu (**Transient**) yang menyumbang **lebih dari 56% total pesanan**, tetapi di sisi lain menjadi sarang utama tumpukan volume pembatalan massal (mencapai **30.5% Cancellation Rate**). Kemudahan fitur pembatalan gratis memicu fenomena double-booking massal, terutama pada puncak musim liburan di bulan **Juli dan Agustus.**

**Komitmen Pelanggan**

Analisis data tunggal membuktikan bahwa hotel dapat menyaring profil tamu yang memiliki kepastian kedatangan hampir 100% (anti-batal) melalui sinyal perilaku berikut:
- Fasilitas Parkir (**required_car_parking_spaces**): Tamu yang meminta slot parkir mobil mencatatkan 0% pembatalan mutlak.
- Tamu yang mengajukan banyak Permintaan Khusus (**total_of_special_requests**) dan aktif melakukan Perubahan Detail Pesanan **(booking_changes)** memiliki kedekatan emosional tinggi untuk berkomitmen tetap datang.
- Pemesanan atas nama akun perusahaan/kontrak dinas (**is_company = 1**) sangat aman dengan risiko batal minimum, hanya sekitar 11.4%.

**Geodemografis**

Meskipun volume pembatalan terbesar secara nominal berasal dari pasar domestik Portugal (PRT), **negara Angola (AGO)** menjadi wilayah dengan efisiensi terburuk karena lebih dari setengah pesanannya **(56.6%)** berakhir batal akibat kendala regulasi visa internasional.

></div>
****
## **4. Modelling**

- Model terbaik: XGBoost classifier + class_weight + thresholds 0.54
- best param_space = {'model__subsample': 0.6,
                     'model__scale_pos_weight': 1,
                     'model__reg_lambda': 5.0,
                     'model__reg_alpha': 0.5,
                     'model__n_estimators': 300,
                     'model__min_child_weight': 3,
                     'model__max_depth': 6,
                     'model__learning_rate': 0.1,
                     'model__gamma': 0.1,
                     'model__colsample_bytree': 0.7}
  
- performance metrics:
  
  basexgb MODEL
  - fbeta Train = 68.61%
  - fbeta Test = 64.38%
    
  bestxgb MODEL
  - fbeta Train = 75.88%
  - fbeta Test = 72.13%

- Business Impact: dengan mengimplementasikan model Best XGBoost, hotel berhasil memangkas total kerugian dari 500,010 Euro (tanpa mode) menjadi 443,910 Euro(dengam model). Artinya, kehadiran model ML sukses menyelamatkan uang hotel sebesar 56,100 Euro (Sekitar 950 Juta Rupiah).
- Fitur yang mempengaruhi pembatalan:
  - lead_time : Semakin jauh hari tamu melakukan reservasi, semakin besar ketidakpastian rencana perjalanan mereka, sehingga model mengidentifikasinya sebagai pemicu utama pembatalan.
  - previous_cancellations: Tamu yang tercatat pernah membatalkan pesanan sebelumnya memiliki kecenderungan mutlak untuk kembali membatalkan pesanannya saat ini.
  - adr (Tarif Kamar):  Tamu yang memesan dengan harga tinggi lebih sensitif terhadap harga dan selektif, mereka cenderung mudah membatalkan pesanan jika menemukan hotel kompetitor yang lebih murah sebelum hari H.
  - stay_duration (Durasi Menginap): Durasi menginap yang panjang membawa risiko ketidakpastian agenda yang lebih tinggi dibandingkan tamu jangka pendek.

- Recommendation Untuk Tim Revenue Management(Manajer Pendapatan):
  - **Penerapan Overbooking Terpandu Model**, Integrasikan probabilitas prediksi model Best XGBoost ke dalam sistem Property Management System (PMS). Gunakan batas keputusan (Probability Threshold) optimal hasil kurva tuning sebesar 0.56 untuk menentukan kuota overbooking harian secara dinamis (disarankan berkisar antara 5% - 15% dari kapasitas kamar).
  - **Perbaiki Kebijakan Deposit Jangka Panjang**, Untuk pemesanan yang dibuat jauh-jauh hari (lead_time > 90 hari), matikan opsi No Deposit dan wajibkan skema pembayaran awal.
  - **Audit Agensi**, Lakukan evaluasi berkala terhadap ID agent atau pihak ketiga yang menyumbang angka pembatalan Non-Refundable massal guna membersihkan sistem dari pemesanan spekulatif yang merugikan biro distribusi kamar.
  - **Protokol Verifikasi Manual Kamar Rombongan**, Untuk pemesanan retail individu (Transient) dan rombongan keluarga besar berjumlah 3-4 Orang (yang memiliki risiko batal 34.3%), lakukan prosedur follow-up atau konfirmasi kehadiran otomatis/manual pada H-7 dan H-3 sebelum kedatangan guna memastikan ketersediaan kuota kamar fisik sebelum dialihkan ke tamu walk-in.

****
### Tech Stack
****
- Data Processing & Analysis: Python, pandas, numpy, scipy, statsmodels
- Visualization: matplotlib, seaborn, folium (geospatial heatmap), Tableau Public
- Machine Learning:scikit-learn, XGBoost, imbalanced-learn, SHAP, category-encoders
- Deployment: Streamlit, pickle (model persistence)
****
**Link Dashboard** : https://datastudio.google.com/reporting/df4d54ac-c280-49f0-8d37-87b4a6278714

**Link Streamlit**: https://finpro-ds-it39wabbxb2kztzj9axplu.streamlit.app/

**Link PPT**: https://canva.link/w9l14jn7x9c53uq

****
### Dokumentasi
****
**Dokumentasi Dashboard**
<img width="908" height="682" alt="Screenshot (344)" src="https://github.com/user-attachments/assets/feba1de6-7493-447e-b9dd-a0cccff288d5" />

**Dokumentasi Streamlit** 
<img width="1794" height="780" alt="Screenshot (341)" src="https://github.com/user-attachments/assets/aba10b09-6fc8-4772-80d7-0d10add34504" />
