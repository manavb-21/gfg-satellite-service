import os
from flask import Flask, request, jsonify
import pystac_client
import odc.stac
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from io import BytesIO
import base64
import datetime

app = Flask(__name__)

def calculate_carbon_credits_api(coordinates, start_date='2023-01-01', end_date='2024-12-30', max_cloud_cover=20):
    """
    Calculate carbon credits based on satellite imagery analysis
    """
    # Carbon sequestration rates by land type (tonnes CO2/hectare/year)
    carbon_rates = {
        0: 0.1,  # Built-up areas
        1: 1.5,  # Low vegetation
        2: 5.0   # High vegetation (trees)
    }

    # Land type names
    land_types = {
        0: 'Built-up',
        1: 'Low Vegetation',
        2: 'High Vegetation (Trees)'
    }

    print(f"Connecting to Satellite API for coordinates: {coordinates}")

    # Calculate bounding box from coordinates
    lons = [coord[0] for coord in coordinates]
    lats = [coord[1] for coord in coordinates]
    bbox = [min(lons), min(lats), max(lons), max(lats)]

    print(f"Searching for images in {start_date}/{end_date} with <{max_cloud_cover}% clouds...")

    # Connect to satellite API
    catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")

    # Search for images
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}}
    )

    items = search.item_collection()
    print(f"Found {len(items)} images.")

    if len(items) == 0:
        raise Exception("No images found. Please increase MAX_CLOUD_COVER to 50 or widen DATE_RANGE.")

    # Use the least cloudy image
    selected_item = min(items, key=lambda i: i.properties['eo:cloud_cover'])
    print(f"Selected image from date: {selected_item.datetime} (Cloud cover: {selected_item.properties['eo:cloud_cover']}%)")

    print("Downloading data (Red, Green, Blue, NIR bands)...")

    try:
        data = odc.stac.load(
            [selected_item],
            bands=["red", "green", "blue", "nir"],
            bbox=bbox,
            resolution=10  # 10 meters per pixel
        ).isel(time=0)
    except Exception as e:
        raise Exception(f"Error downloading data: {e}")

    # Check if data is empty
    if data.red.size == 0:
        raise Exception("Downloaded data is empty. Your BBOX might be too small for Sentinel-2 resolution.")

    print("Calculating Vegetation Index (NDVI)...")

    # Calculate NDVI
    nir = data.nir.astype(float)
    red = data.red.astype(float)
    ndvi = (nir - red) / (nir + red)
    ndvi = ndvi.fillna(0)

    print("Running AI Classification...")

    # Prepare data for K-Means
    h, w = ndvi.shape
    X = ndvi.values.reshape((-1, 1))  # Flatten to 1D array

    # K-Means Clustering (3 Classes: Built, Low Veg, High Veg)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    # Sort labels so 0=LowNDVI, 2=HighNDVI
    centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(centers)
    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
    sorted_labels = np.vectorize(mapping.get)(labels)

    # Reshape back to image
    classification = sorted_labels.reshape((h, w))

    print("Generating plots...")

    # Create plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1: True Color
    # Brighten the image (Sentinel data is raw and dark)
    rgb = np.dstack((data.red, data.green, data.blue))
    p2, p98 = np.percentile(rgb, (2, 98))
    rgb_bright = np.clip((rgb - p2) / (p98 - p2), 0, 1)

    ax[0].imshow(rgb_bright)
    ax[0].set_title("Satellite Image (True Color)")
    ax[0].axis('off')

    # Plot 2: Analysis
    cmap = plt.cm.get_cmap("RdYlGn", 3)  # Red-Yellow-Green
    im = ax[1].imshow(classification, cmap=cmap)
    ax[1].set_title("Vegetation Analysis (Green=Trees)")
    ax[1].axis('off')

    # Save plots to base64 strings
    satellite_img_buffer = BytesIO()
    analysis_img_buffer = BytesIO()

    # Save satellite image
    ax[0].figure.savefig(satellite_img_buffer, format='png', bbox_inches='tight', dpi=100)
    satellite_img_buffer.seek(0)
    satellite_img_str = base64.b64encode(satellite_img_buffer.read()).decode()

    # Save analysis image
    ax[1].figure.savefig(analysis_img_buffer, format='png', bbox_inches='tight', dpi=100)
    analysis_img_buffer.seek(0)
    analysis_img_str = base64.b64encode(analysis_img_buffer.read()).decode()

    plt.close(fig)

    print("Calculating land composition and carbon potential...")

    # Calculate area for each land type
    area_per_pixel = 100  # 10m x 10m = 100 m² per pixel
    unique, counts = np.unique(classification, return_counts=True)

    composition = []
    total_area_hectares = 0
    total_carbon_tonnes = 0

    for land_type, count in zip(unique, counts):
        area_hectares = (count * area_per_pixel) / 10000  # Convert m² to hectares
        annual_carbon = area_hectares * carbon_rates.get(int(land_type), 0)

        composition.append({
            'land_type_id': int(land_type),
            'land_type_name': land_types.get(int(land_type), f'Unknown ({int(land_type)})'),
            'area_hectares': float(area_hectares),
            'annual_carbon_tonnes': float(annual_carbon)
        })

        total_area_hectares += area_hectares
        total_carbon_tonnes += annual_carbon

    # Calculate NDVI statistics
    mean_ndvi = float(np.nanmean(ndvi.values))
    std_ndvi = float(np.nanstd(ndvi.values))

    # Prepare results
    results = {
        'property_boundary': coordinates,
        'assessment_period': {
            'start_date': start_date,
            'end_date': end_date
        },
        'land_composition': composition,
        'total_property_area_hectares': total_area_hectares,
        'total_annual_carbon_sequestration_tonnes': total_carbon_tonnes,
        'vegetation_health': {
            'mean_ndvi': mean_ndvi,
            'std_ndvi': std_ndvi
        },
        'timestamp': 'Calculating...',
        'images': {
            'satellite': f"data:image/png;base64,{satellite_img_str}",
            'analysis': f"data:image/png;base64,{analysis_img_str}"
        }
    }

    results['timestamp'] = datetime.datetime.now().isoformat()

    return results

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'satellite-verification-api'
    })

@app.route('/calculate', methods=['POST'])
def calculate():
    """Calculate carbon credits based on satellite imagery analysis"""
    try:
        data = request.get_json()
        coordinates = data.get('coordinates', [])
        start_date = data.get('startDate', '2023-01-01')
        end_date = data.get('endDate', '2024-12-30')

        if not coordinates:
            return jsonify({'error': 'Coordinates are required'}), 400

        if len(coordinates) < 3:
            return jsonify({'error': 'At least 3 coordinate pairs are required to form a polygon'}), 400

        # Calculate carbon credits
        results = calculate_carbon_credits_api(coordinates, start_date, end_date)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5001)  # Changed debug to False for production, port 5001 to avoid conflict