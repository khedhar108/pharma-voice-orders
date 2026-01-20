from typing import List, Dict
import streamlit as st
from .manufacturer_db import ManufacturerDB

class OrderQueue:
    def __init__(self):
        # Initialize session state for orders if not exists
        if 'orders' not in st.session_state:
            st.session_state.orders = []
    
    def add_order(self, order: Dict):
        """
        Add a new order to the queue.
        Order dict should contain: {'medicine': str, 'quantity': str, 'dosage': str}
        """
        st.session_state.orders.append(order)
        
    def get_all_orders(self) -> List[Dict]:
        return st.session_state.orders
    
    def clear_queue(self):
        st.session_state.orders = []
        
    def get_grouped_orders(self, db: ManufacturerDB) -> Dict[str, List[Dict]]:
        """Group all current orders by manufacturer."""
        return db.get_orders_by_manufacturer(st.session_state.orders)
