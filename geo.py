# geo.py
import requests

def get_user_location():
    try:
        response = requests.get("https://ipapi.co/json/", timeout=3)
        if response.status_code == 429:
            return {
                "city": "Toulouse",
                "region": "Occitanie",
                "latitude": 43.6045,
                "longitude": 1.444
            }

        data = response.json()
        city = data.get("city")
        region = data.get("region")
        latitude = data.get("latitude")
        longitude = data.get("longitude")

        if city:
            print(f" Localisation détectée automatiquement : {city}, {region}")
            return {
                "city": city,
                "region": region,
                "latitude": latitude,
                "longitude": longitude
            }
        else:
            print(" Aucune ville détectée → Ville par défaut : Toulouse")
            return {
                "city": "Toulouse",
                "region": "Occitanie",
                "latitude": 43.6045,
                "longitude": 1.444
            }

    except Exception as e:
        print(" Erreur lors de la récupération de la géolocalisation :", e)
        return {
            "city": "Toulouse",
            "region": "Occitanie",
            "latitude": 43.6045,
            "longitude": 1.444
        }
