import re
import yaml
from dataclasses import dataclass, is_dataclass, replace
from typing import get_args, get_origin, get_type_hints, List

import mlflow
from mlflow.exceptions import RestException
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession


@dataclass
class BaseConfig:
    """Base config to extend

    NOTE: This class recursively builds dataclass objects on init, but it is not
    fully featured. Specifically, it only handles fields with a dataclass type or
    List[dataclass] type. More complex types like Dict or Union that incorporate
    dataclasses will not init as expected.

    NOTE: Could instead use `dacite`, but that would add a dependency.
    """

    @classmethod
    def from_yaml(cls, filename: str):
        """Initialize config object from yaml file"""
        with open(filename, "r") as f:
            config = yaml.safe_load(f)
        return cls(**config)

    def __post_init__(self):
        """Recursively build dataclass objects"""
        type_hints = get_type_hints(self)
        for k, v in type_hints.items():
            attr_value = getattr(self, k)
            if attr_value is None:
                continue
            # dataclass type
            if is_dataclass(v):
                new_val = v(**attr_value)
                setattr(self, k, new_val)
            # list of dataclass type
            elif get_origin(v) is list:
                args = get_args(v)
                if len(args) > 0 and is_dataclass(args[0]):
                    new_val = [args[0](**list_item) for list_item in attr_value]
                    setattr(self, k, new_val)


# Generic types


@dataclass
class UCTable(BaseConfig):
    catalog: str
    schema: str
    table: str

    @property
    def path(self) -> str:
        """Returns dot separated path to table"""
        return f"{self.catalog}.{self.schema}.{self.table}"


@dataclass
class UCVolume(BaseConfig):
    """Represents a UC volume or file in a volume"""

    catalog: str
    schema: str
    volume: str
    filename: str = None

    @property
    def path(self) -> str:
        """Returns the dbfs path to the volume/file"""
        path = f"dbfs:/Volumes/{self.catalog}/{self.schema}/{self.volume}/"
        if self.filename is not None:
            path += self.filename
        return path

    @property
    def dot_path(self) -> str:
        """Generate the dot separated path to the volume (ignores filename)

        Can be used to access the volume in spark SQL.
        """
        return f"{self.catalog}.{self.schema}.{self.volume}"

    @property
    def table_path(self) -> str:
        """Generate a table path with the same name as the filename

        Can be used to create a table with the same data as what goes into the
        volume for easy querying and debugging.
        """
        stripped_filename = re.sub(r"\.[a-z]+$", "", self.filename)
        return f"{self.catalog}.{self.schema}.{stripped_filename}"

    @property
    def s3_path(self) -> str:
        """Generate the raw s3 path to the volume/file"""
        spark = SparkSession.builder.getOrCreate()
        s3_location = (
            spark.sql(f"DESCRIBE VOLUME {self.dot_path}").first().storage_location
        )
        if self.filename is not None:
            s3_location = f"{s3_location}/{self.filename}"
        return s3_location


@dataclass
class VSIndex(BaseConfig):
    """Represents a VS Index"""

    catalog: str
    schema: str
    delta_sync_table: str
    index_name: str
    vs_endpoint: str

    @property
    def path(self) -> str:
        """Generate the dot separated path to VS index
        """
        return f"{self.catalog}.{self.schema}.{self.index_name}"

    @property
    def sync_table_path(self) -> str:
        """Generate the dot separated path to Delta table used in the sync to the VS index
        """
        return f"{self.catalog}.{self.schema}.{self.delta_sync_table}"

    @property
    def endpoint(self) -> str:
        """returns name of endpoint that the VS index is registered under"""
        return f"{self.vs_endpoint}"


@dataclass
class UCModel(BaseConfig):
    catalog: str
    schema: str
    model_name: str
    model_version: int = None
    model_alias: str = None
    # This is only needed for the news base models that were trained in a different workspace.
    # We can't lookup the checkpoint file for these from the model name/version because the run
    # lives in a different workspace.
    _s3_checkpoint: str = None

    def __post_init__(self):
        # Should only specify one of `model_version` or `model_alias`
        if self.model_version is not None and self.model_alias is not None:
            raise ValueError(
                "Please specify only one of `model_version` and `model_alias`"
            )
        # Infer model version from model alias
        # NOTE: we don't infer model alias from model version, because there can multiple
        # aliases attached to a version.
        if self.model_alias is not None:
            mlflow.set_registry_uri("databricks-uc")
            client = mlflow.MlflowClient()
            try:
                mv = client.get_model_version_by_alias(self.path, self.model_alias)
                self.model_version = mv.version
            except RestException as exc:
                # Don't fail if model does not exist yet
                if exc.get_http_status_code() != 404:
                    raise

        super().__post_init__()

    @property
    def path(self) -> str:
        """Returns dot separated path to the model"""
        return f"{self.catalog}.{self.schema}.{self.model_name}"

    @property
    def uri(self) -> str:
        """Get the model URI"""
        if self.model_alias is not None:
            return f"models:/{self.path}@{self.model_alias}"
        elif self.model_version is not None:
            return f"models:/{self.path}/{self.model_version}"
        else:
            raise ValueError(
                "Need to specify one of `model_version` and `model_alias` for a URI"
            )

    @property
    def s3_checkpoint(self):
        """Get the s3 checkpoint path to train on an existing base model"""
        if self._s3_checkpoint is not None:
            return self._s3_checkpoint
        # fetch the run
        mlflow.set_registry_uri("databricks-uc")
        client = mlflow.MlflowClient()
        if self.model_version is not None:
            mv = client.get_model_version(self.path, self.model_version)
        elif self.model_alias is not None:
            mv = client.get_model_version_by_alias(self.path, self.model_alias)
        else:
            raise ValueError(
                "Must specify `model_version` or `model_alias` to get checkpoint"
            )
        run = client.get_run(mv.run_id)
        # fetch symlink to latest checkpoint
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        paths = [
            p
            for p in dbutils.fs.ls(run.data.params["save_folder"])
            if p.name.endswith("symlink")
        ]
        return paths[0].path


@dataclass
class DeployConfig(BaseConfig):
    endpoint_name: str
    dev_model: UCModel
    image_source: UCVolume
    image_table: UCTable
    user_table: UCTable
    user_online_table: UCTable
    vs_index: VSIndex   
                
