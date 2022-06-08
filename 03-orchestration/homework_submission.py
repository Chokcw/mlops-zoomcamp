import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
def main(date=None):

    if date == None:
        date = datetime.today()

    else:
        date = datetime.strptime(date, '%Y-%m-%d')

    train_file_date = (date + relativedelta(months=-2)).strftime('%Y-%m')
    val_file_date = (date + relativedelta(months=-1)).strftime('%Y-%m')

    train_path: str = f'./data/fhv_tripdata_{train_file_date}.parquet' 
    val_path: str = f'./data/fhv_tripdata_{val_file_date}.parquet'

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical).result()

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False).result()

    # train the model
    lr, dv = train_model(df_train_processed, categorical)
    with open(f"models/model-{date.strftime('%Y-%m-%d')}.bin", "wb") as f_out:
            pickle.dump(dv, f_out)
    run_model(df_val_processed, categorical, dv, lr)


from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(), # to specify it to run like a normal python script, instead of spinning up a new docker image or kubes pod
    tags=["ml"]
)

main(date="2021-08-15")

# Q1 
# Q2 - 11.637
# Q3 - 13,000 bytes
# Q4 - 0 9 15 * *
# Q5 - 4
# Q6 - prefect work-queue ls