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

🔹Intrusion Detection for Electric Vehicle Charging Systems (EVCS)
Published in Algorithms, 2023.
Focuses on machine‐learning based IDS applied to EVCS ecosystem (IoT component)

🔹Cyber defense in OCPP for EV charging security risks
Published 2025 in Information and Computer Security (Springer).

🔹Federated detection of Open Charge Point Protocol 1.6 cyberattacks
Published 2025.
Presents an FL (federated learning)-based IDS architecture for OCPP 1.6 traffic.

🔹Explainable Deep Learning for Cyber Attack Detection in Electric Vehicle Charging Stations
Published (2023).
Focuses on deep learning + explainability applied in EVCS intrusion detection.

🔹Buedi, Emmanuel Dana, et al. "Enhancing EV Charging Station Security Using a Multi-Dimensional Dataset: CICEVSE2024." IFIP Annual Conference on Data and Applications Security and Privacy. 

🔹“EVSE Dataset 2024 | Datasets | Research | Canadian Institute for Cybersecurity.” University of New Brunswick, https://www.unb.ca/cic/datasets/evse-dataset-2024.html. 

✨Contributor
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Sweta Ghosh (@SwetaGhosh119)

Netaji Subhas Engineering College

Email: britneyspears9246@gmail.com
