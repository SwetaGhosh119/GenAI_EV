# GenAI_EV
Intrusion Detection Model for Electric Vehicles (GenAI Approach)🚗🪫🔌⚡️⚠️🔒

⚡ Temporal Secure Intrusion Detection System for Electric Vehicle Charging (OCPP 1.6)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📖 Overview
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

This project implements a temporal secure Intrusion Detection System (IDS) for Electric Vehicle Charging Stations (EVCS) communicating via the Open Charge Point Protocol (OCPP 1.6).
The system leverages Machine Learning (ML), Deep Learning (CNN/LSTM), Wasserstein Conditional GAN (WCGAN) for data generation, and an integrated Gen-AI Chatbot to enhance security awareness and interpretability.

The CIC EV Charger Attack Dataset 2024 (CICEVSE2024) is the core dataset utilized in this project. It provides an extensive representation of normal and malicious activities in Electric Vehicle Supply Equipment (EVSE) systems. The dataset records crucial aspects such as power usage, network communication data, and host-level event logs including Hardware Performance Counters (HPC) and Kernel Events.

The dataset is organized into three primary segments:

Network Traffic: Contains .pcap files and processed .csv files for two chargers—EVSE-A and EVSE-B.

Host Events: Includes detailed logs of HPC and Kernel Events for EVSE-B, captured under both normal and attack conditions.

Power Consumption: Features readings showing variations in power usage between standard operation and compromised states of EVSE-B.

This dataset enables behavioral profiling, anomaly detection, and performance evaluation of EV charging systems. It supports both statistical and machine learning-based methods, making it a key asset for uncovering and analyzing vulnerabilities in EV infrastructure.

🚀 Objectives
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

✅️Detect cyberattacks and anomalies in EV charging communication traffic.

✅️Analyze unauthorized access, DoS, and malware attacks using OCPP datasets.

✅️Build a temporal secure IDS using CNN, LSTM, and WCGAN.

✅️Create a Gen-AI chatbot to explain, visualize, and recommend responses to security events.

🧩 System Architecture

Modules:
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

⮞Data Preprocessing & Cleaning – from OCPP event logs

⮞ML-based IDS Models – Random Forest, XGBoost, SVM, etc.

⮞CNN/CV-based Intrusion Detection – visual temporal representation of traffic

⮞WCGAN Data Augmentation – generate synthetic attack data

⮞Temporal LSTM Security Model – detect sequential anomalies

⮞Gen-AI Chatbot – explain IDS results and assist EVCS operators

🧠 Tech Stack
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

|  Component        |    Tools / Libraries                |
|:-----------------:|:-----------------------------------:|
|  Language	        |    Python 3.10+                     |
|  ML / DL          |    Scikit-learn, TensorFlow, PyTorch|
|  Data Handling    |	    Pandas, NumPy                   |
|  Visualization    |	   Matplotlib, Seaborn              |  
|  Generative Models|    	Wasserstein Conditional GAN     |
|  Chatbot	        | HuggingFace Transformers / LangChain|
|  Frontend         |  Streamlit / Flask + React          |
|Protocol Simulated |  OCPP 1.6 (Central System – EVCS)   |

📈 Evaluation Metrics
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔶Accuracy

🔶Precision, Recall, F1-score

🔶ROC-AUC Curve

🔶Confusion Matrix

🔶Temporal anomaly metrics (for LSTM models)

📚 References
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🔹Hussain, R., et al. “Cybersecurity in Electric Vehicle Charging Infrastructure: Challenges and Solutions.” IEEE Transactions on Smart Grid, 2022.
https://doi.org/10.1109/TSG.2022.3142023

🔹Mohamed, N., et al. “A Deep Learning-Based Intrusion Detection System for Electric Vehicle Charging Stations.” IEEE Access, 2021.
https://doi.org/10.1109/ACCESS.2021.3076372

🔹Hafeez, I., et al. “Electric Vehicle Charging Infrastructure: Vulnerabilities, Attacks, and Countermeasures.” IEEE Transactions on Transportation Electrification, 2020.
https://doi.org/10.1109/TTE.2020.3037489

🔹Dutta, D., et al. “Temporal Convolutional Neural Networks for Network Intrusion Detection.” Computer Networks, 2022.
https://doi.org/10.1016/j.comnet.2022.109180

🔹Zhang, J., et al. “Wasserstein GAN-Based Data Augmentation for Imbalanced Intrusion Detection.” IEEE Access, 2020.
https://doi.org/10.1109/ACCESS.2020.2966011

🔹Wu, T., et al. “A Federated Learning-Based Intrusion Detection System for Electric Vehicle Networks.” IEEE Internet of Things Journal, 2023.
https://doi.org/10.1109/JIOT.2023.3235402

🔹Open Charge Alliance. “OCPP 1.6 Specification.”
https://www.openchargealliance.org/protocols/ocpp-16/

🔹ISO 15118-20 (Vehicle-to-Grid Communication Interface) – Defines secure communication between EVs and CSMS.
https://www.iso.org/standard/82769.html

🔹Buedi, Emmanuel Dana, et al. "Enhancing EV Charging Station Security Using a Multi-Dimensional Dataset: CICEVSE2024." IFIP Annual Conference on Data and Applications Security and Privacy. 

🔹“EVSE Dataset 2024 | Datasets | Research | Canadian Institute for Cybersecurity.” University of New Brunswick, https://www.unb.ca/cic/datasets/evse-dataset-2024.html. 

✨Contributor
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Sweta Ghosh (@SwetaGhosh119)

Netaji Subhas Engineering College

Email: britneyspears9246@gmail.com
