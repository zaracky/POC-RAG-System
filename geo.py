# geo.py
import requests

def get_user_location():
    try:
        response = requests.get("https://ipapi.co/json/")
        data = response.json()
        return {
            "city": data.get("city"),
            "region": data.get("region"),
            "latitude": data.get("latitude"),
            "longitude": data.get("longitude")
        }
    except Exception as e:
        print("Erreur lors de la récupération de la géolocalisation IP :", e)
        return None
