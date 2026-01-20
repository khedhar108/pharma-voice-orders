import pandas as pd
import io

class ExcelExporter:
    @staticmethod
    def _prepare_dataframe(orders: list, db=None) -> pd.DataFrame:
        """Helper to prepare enriched DataFrame."""
        if not orders:
            return pd.DataFrame()

        # Enrich data if DB is provided
        enriched_orders = []
        for order in orders:
            row = order.copy()
            if db:
                mfr_info = db.get_manufacturer_by_medicine(order['medicine'])
                if mfr_info:
                    row['Manufacturer'] = mfr_info['name']
                    row['Standardized Medicine'] = mfr_info['medicine_match']
                else:
                    row['Manufacturer'] = "Unknown"
                    row['Standardized Medicine'] = "-"
            enriched_orders.append(row)

        df = pd.DataFrame(enriched_orders)
        
        # Rename columns for better readability
        column_map = {
            "medicine": "Medicine Name (Extracted)",
            "quantity": "Quantity",
            "dosage": "Dosage",
            "original_segment": "Raw Voice Segment",
            "Manufacturer": "Manufacturer",
            "Standardized Medicine": "Standardized Name"
        }
        df = df.rename(columns=column_map)
        
        # Reorder columns if possible
        desired_order = [
            "Manufacturer", 
            "Standardized Name", 
            "Medicine Name (Extracted)", 
            "Quantity", 
            "Dosage", 
            "Raw Voice Segment"
        ]
        
        cols_to_keep = [c for c in desired_order if c in df.columns]
        remaining = [c for c in df.columns if c not in cols_to_keep]
        
        return df[cols_to_keep + remaining]

    @staticmethod
    def export(orders: list, db=None) -> bytes:
        """Convert list of order dicts to Excel bytes."""
        df = ExcelExporter._prepare_dataframe(orders, db)
        if df.empty:
            return None
            
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Orders')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Orders']
            for idx, col in enumerate(df.columns):
                max_len = max(
                    df[col].astype(str).map(len).max(),
                    len(col)
                ) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_len, 50)
            
        return output.getvalue()

    @staticmethod
    def export_csv(orders: list, db=None) -> str:
        """Convert list of order dicts to CSV string."""
        df = ExcelExporter._prepare_dataframe(orders, db)
        if df.empty:
            return ""
        return df.to_csv(index=False).encode('utf-8')
