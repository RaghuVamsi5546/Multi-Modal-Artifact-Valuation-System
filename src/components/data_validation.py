import pandas as pd
import os
from src.utils.common import *

from src.entity import DataValidationConfig
from sklearn.model_selection import train_test_split

class DataValidation:
    def __init__(self,config:DataValidationConfig,schema):
        self.config=config
        self.schema_path=schema
        
    def validate_columns(self):
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
    
            if status == True:
                X=df.drop(columns=['preservation_score'])
                y=df['preservation_score']
                X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
                X_train,X_valid,y_train,y_valid=train_test_split(X_train,y_train,test_size=0.2,random_state=42)

                train_df = pd.concat([X_train, y_train], axis=1)
                valid_df = pd.concat([X_valid, y_valid], axis=1)
                test_df = pd.concat([X_test, y_test], axis=1)

                os.makedirs(self.config.root_dir, exist_ok=True)
                train_df.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
                valid_df.to_csv(os.path.join(self.config.root_dir, "validation.csv"), index=False)
                test_df.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)
                
            return status,train_df,valid_df,test_df

        except Exception as e:
            raise e