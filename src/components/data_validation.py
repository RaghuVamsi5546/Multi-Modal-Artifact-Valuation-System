import pandas as pd

from src.utils.common import *

from src.entity import DataValidationConfig

class DataValidation:
    def __init__(self,config:DataValidationConfig,schema):
        self.config=config
        self.schema_path=schema
        
    def validate_columns(self)->bool:
        try:
            status=True
            messages=[]
            df=pd.read_csv(self.config.local_file)

            self.schema=read_yaml(self.schema_path) 

            expected_columns=list(self.schema.keys())
            data_columns=df.columns.tolist()

            missing_cols = [col for col in expected_columns if col not in data_columns]
            if missing_cols:
                status = False
                messages.append(f"Missing columns: {missing_cols}")

            def normalize_dtype(dtype_str):
                if "float" in dtype_str:
                    return "float"
                if "int" in dtype_str:
                    return "int"
                return dtype_str
            
            for col, col_type in self.schema.items():
                if col in df.columns:
                    actual_dtype = normalize_dtype(str(df[col].dtype))
                    if actual_dtype != col_type:
                        status = False
                        print(actual_dtype,col_type)    
            with open(self.config.validation_status_path, 'w') as f:
                f.write(str(status))
            return status
        except Exception as e:
            raise e