import streamlit as st
import requests

st.title("EV-IDS Intrusion Detection Demo")

# Create 10 inputs
irq_softirq_exit = st.number_input("irq_softirq_exit", 0.0)
irq_softirq_entry = st.number_input("irq_softirq_entry", 0.0)
irq_softirq_raise = st.number_input("irq_softirq_raise", 0.0)
kmem_kmem_cache_free = st.number_input("kmem_kmem_cache_free", 0.0)
kmem_kmem_cache_alloc = st.number_input("kmem_kmem_cache_alloc", 0.0)
net_netif_rx = st.number_input("net_netif_rx", 0.0)
net_netif_rx_ni_exit = st.number_input("net_netif_rx_ni_exit", 0.0)
net_netif_rx_ni_entry = st.number_input("net_netif_rx_ni_entry", 0.0)
rpm_rpm_usage = st.number_input("rpm_rpm_usage", 0.0)
rpm_rpm_resume = st.number_input("rpm_rpm_resume", 0.0)

if st.button("Predict"):
    payload = {
        "features": [
            irq_softirq_exit,
            irq_softirq_entry,
            irq_softirq_raise,
            kmem_kmem_cache_free,
            kmem_kmem_cache_alloc,
            net_netif_rx,
            net_netif_rx_ni_exit,
            net_netif_rx_ni_entry,
            rpm_rpm_usage,
            rpm_rpm_resume
        ]
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    st.write("🔍 Prediction Result:")
    st.json(response.json())
