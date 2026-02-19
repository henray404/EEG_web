alur proses sinyal eeg



di file edf:



resting

thinking

acting

typing



pipeline:



Dari data Raw -> Preprocessing (Bandpass/ Notch Filter -> ICA filter) -> 1 file diekstrak per flag dan per subject, struktur folder: (id/skenario/flag) dalam 1 folder skenario ada 4 folder task -> Ambil Channel tertentu -> Bagi jadi subband -> Lakukan fitur extraksi\* -> eksperimen



opsi eksperimen:



(karena sinyal eeg itu relative maka ga boleh membandingkan 1 sinyal eeg yg baru di FE )



**baseline**

\- ambil delta perbedaan dari transisi antar task lalu bandingkan als vs normal



**opsi lain**

\- lakukan normalisasi dulu per subjek sebelum dibandingkan antar class





