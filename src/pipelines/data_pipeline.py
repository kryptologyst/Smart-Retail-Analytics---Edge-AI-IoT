"""Data pipeline and sensor simulation for Smart Retail Analytics."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging


@dataclass
class RetailSensorData:
    """Data class for retail sensor measurements."""
    customer_id: str
    timestamp: float
    time_in_store: float  # minutes
    sections_visited: int
    items_touched: int
    interaction_time: float  # seconds
    dwell_time: float  # seconds
    footfall_count: int
    temperature: float  # Celsius
    humidity: float  # percentage
    noise_level: float  # dB
    purchase_likelihood: Optional[float] = None


class RetailDataGenerator:
    """Generate synthetic retail analytics data for training and testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize data generator.
        
        Args:
            seed: Random seed for reproducibility.
        """
        np.random.seed(seed)
        self.logger = logging.getLogger(__name__)
    
    def generate_customer_data(
        self,
        n_samples: int = 1000,
        store_size: str = "medium"
    ) -> List[RetailSensorData]:
        """Generate synthetic customer behavior data.
        
        Args:
            n_samples: Number of customer samples to generate.
            store_size: Store size affecting behavior patterns ("small", "medium", "large").
            
        Returns:
            List of RetailSensorData objects.
        """
        # Store size parameters
        store_params = {
            "small": {"max_sections": 5, "avg_time": 8, "std_time": 3},
            "medium": {"max_sections": 10, "avg_time": 15, "std_time": 5},
            "large": {"max_sections": 15, "avg_time": 25, "std_time": 8}
        }
        
        params = store_params.get(store_size, store_params["medium"])
        
        data = []
        for i in range(n_samples):
            # Generate customer behavior patterns
            time_in_store = max(1, np.random.normal(params["avg_time"], params["std_time"]))
            sections_visited = min(
                params["max_sections"],
                max(1, np.random.poisson(params["max_sections"] * 0.6))
            )
            items_touched = max(0, np.random.poisson(3))
            interaction_time = max(5, np.random.exponential(30))
            dwell_time = max(10, np.random.exponential(60))
            
            # Environmental factors
            footfall_count = max(1, np.random.poisson(20))
            temperature = np.random.normal(22, 2)  # Store temperature
            humidity = np.random.normal(45, 10)  # Store humidity
            noise_level = np.random.normal(65, 10)  # Store noise level
            
            # Generate purchase likelihood based on behavior patterns
            purchase_likelihood = self._calculate_purchase_likelihood(
                time_in_store, sections_visited, items_touched, interaction_time
            )
            
            sensor_data = RetailSensorData(
                customer_id=f"customer_{i:04d}",
                timestamp=np.random.uniform(0, 86400),  # Random time in day
                time_in_store=time_in_store,
                sections_visited=sections_visited,
                items_touched=items_touched,
                interaction_time=interaction_time,
                dwell_time=dwell_time,
                footfall_count=footfall_count,
                temperature=temperature,
                humidity=humidity,
                noise_level=noise_level,
                purchase_likelihood=purchase_likelihood
            )
            
            data.append(sensor_data)
        
        self.logger.info(f"Generated {n_samples} customer data samples")
        return data
    
    def _calculate_purchase_likelihood(
        self,
        time_in_store: float,
        sections_visited: int,
        items_touched: int,
        interaction_time: float
    ) -> float:
        """Calculate purchase likelihood based on customer behavior.
        
        Args:
            time_in_store: Time spent in store (minutes).
            sections_visited: Number of store sections visited.
            items_touched: Number of items touched.
            interaction_time: Time spent interacting with items (seconds).
            
        Returns:
            Purchase likelihood score between 0 and 1.
        """
        # Weighted factors for purchase likelihood
        time_factor = min(1.0, time_in_store / 20)  # Normalize to 20 minutes
        sections_factor = min(1.0, sections_visited / 8)  # Normalize to 8 sections
        items_factor = min(1.0, items_touched / 5)  # Normalize to 5 items
        interaction_factor = min(1.0, interaction_time / 60)  # Normalize to 60 seconds
        
        # Weighted combination with some randomness
        base_likelihood = (
            0.3 * time_factor +
            0.2 * sections_factor +
            0.3 * items_factor +
            0.2 * interaction_factor
        )
        
        # Add some noise
        noise = np.random.normal(0, 0.1)
        likelihood = np.clip(base_likelihood + noise, 0, 1)
        
        return likelihood
    
    def to_dataframe(self, data: List[RetailSensorData]) -> pd.DataFrame:
        """Convert sensor data to pandas DataFrame.
        
        Args:
            data: List of RetailSensorData objects.
            
        Returns:
            DataFrame with sensor data.
        """
        records = []
        for sensor_data in data:
            record = {
                "customer_id": sensor_data.customer_id,
                "timestamp": sensor_data.timestamp,
                "time_in_store": sensor_data.time_in_store,
                "sections_visited": sensor_data.sections_visited,
                "items_touched": sensor_data.items_touched,
                "interaction_time": sensor_data.interaction_time,
                "dwell_time": sensor_data.dwell_time,
                "footfall_count": sensor_data.footfall_count,
                "temperature": sensor_data.temperature,
                "humidity": sensor_data.humidity,
                "noise_level": sensor_data.noise_level,
                "purchase_likelihood": sensor_data.purchase_likelihood
            }
            records.append(record)
        
        return pd.DataFrame(records)


class RetailDataProcessor:
    """Process and prepare retail data for machine learning."""
    
    def __init__(self):
        """Initialize data processor."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = [
            "time_in_store", "sections_visited", "items_touched",
            "interaction_time", "dwell_time", "footfall_count",
            "temperature", "humidity", "noise_level"
        ]
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training.
        
        Args:
            df: DataFrame with retail data.
            feature_columns: List of feature column names.
            
        Returns:
            Tuple of (features, labels) arrays.
        """
        if feature_columns is None:
            feature_columns = self.feature_columns
        
        # Extract features
        X = df[feature_columns].values
        
        # Create binary labels based on purchase likelihood threshold
        y = (df["purchase_likelihood"] > 0.5).astype(int).values
        
        self.logger.info(f"Prepared features shape: {X.shape}, labels shape: {y.shape}")
        return X, y
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train, validation, and test sets.
        
        Args:
            X: Feature array.
            y: Label array.
            test_size: Proportion of data for testing.
            validation_size: Proportion of data for validation.
            random_state: Random seed.
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        self.logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def normalize_features(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Normalize features using StandardScaler.
        
        Args:
            X_train: Training features.
            X_val: Validation features.
            X_test: Test features.
            
        Returns:
            Tuple of normalized feature arrays.
        """
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Transform validation and test data
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        X_test_scaled = self.scaler.transform(X_test) if X_test is not None else None
        
        self.logger.info("Features normalized using StandardScaler")
        return X_train_scaled, X_val_scaled, X_test_scaled


class StreamingDataSimulator:
    """Simulate real-time streaming data from retail sensors."""
    
    def __init__(self, data_generator: RetailDataGenerator):
        """Initialize streaming simulator.
        
        Args:
            data_generator: RetailDataGenerator instance.
        """
        self.data_generator = data_generator
        self.logger = logging.getLogger(__name__)
    
    def generate_streaming_data(
        self,
        duration_seconds: int = 3600,
        sampling_rate: float = 1.0
    ) -> List[RetailSensorData]:
        """Generate streaming data for specified duration.
        
        Args:
            duration_seconds: Duration of streaming data in seconds.
            sampling_rate: Samples per second.
            
        Returns:
            List of streaming sensor data.
        """
        n_samples = int(duration_seconds * sampling_rate)
        streaming_data = []
        
        for i in range(n_samples):
            # Generate single customer data point
            customer_data = self.data_generator.generate_customer_data(n_samples=1)[0]
            
            # Update timestamp to simulate real-time
            customer_data.timestamp = i / sampling_rate
            
            streaming_data.append(customer_data)
            
            # Simulate some delay if needed
            if i % 100 == 0:
                self.logger.info(f"Generated {i+1}/{n_samples} streaming samples")
        
        return streaming_data
    
    def create_ring_buffer(self, buffer_size: int = 100) -> List[RetailSensorData]:
        """Create a ring buffer for streaming data.
        
        Args:
            buffer_size: Maximum size of the ring buffer.
            
        Returns:
            Empty ring buffer list.
        """
        return [None] * buffer_size
    
    def add_to_buffer(
        self,
        buffer: List[RetailSensorData],
        new_data: RetailSensorData
    ) -> List[RetailSensorData]:
        """Add new data to ring buffer.
        
        Args:
            buffer: Ring buffer list.
            new_data: New sensor data to add.
            
        Returns:
            Updated ring buffer.
        """
        # Shift existing data and add new data at the end
        buffer[:-1] = buffer[1:]
        buffer[-1] = new_data
        
        return buffer
