import 'dart:convert';
import 'dart:async';
import 'package:http/http.dart' as http;

class WeatherInfo {
  final double tempC;
  final double humidity;
  final double rainfallMm;
  final String condition;
  final String city;

  WeatherInfo({
    required this.tempC,
    required this.humidity,
    required this.rainfallMm,
    required this.condition,
    required this.city,
  });
}

class WeatherService {
  // ✅ If you keep this as placeholder -> it will AUTO USE MOCK
  static const String openWeatherKey = "YOUR_OPENWEATHER_API_KEY";

  static Future<WeatherInfo> fetchByLatLon(double lat, double lon) async {
    // ✅ UI STAGE (MOCK)
    if (openWeatherKey == "YOUR_OPENWEATHER_API_KEY") {
      await Future.delayed(const Duration(milliseconds: 600));
      return WeatherInfo(
        tempC: 28.6,
        humidity: 72,
        rainfallMm: 12.4,
        condition: "Cloudy",
        city: "Anuradhapura",
      );
    }

    // ✅ REAL API (later)
    final uri = Uri.parse(
      "https://api.openweathermap.org/data/2.5/weather"
      "?lat=$lat&lon=$lon&appid=$openWeatherKey&units=metric",
    );

    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception("Weather API error: ${res.statusCode}");
    }

    final data = json.decode(res.body);

    final temp = (data["main"]["temp"] as num).toDouble();
    final humidity = (data["main"]["humidity"] as num).toDouble();
    final condition = (data["weather"]?[0]?["main"] ?? "Unknown").toString();
    final city = (data["name"] ?? "Unknown").toString();

    double rain = 0;
    final rainObj = data["rain"];
    if (rainObj is Map) {
      if (rainObj["1h"] != null) rain = (rainObj["1h"] as num).toDouble();
      if (rainObj["3h"] != null) rain = (rainObj["3h"] as num).toDouble();
    }

    return WeatherInfo(
      tempC: temp,
      humidity: humidity,
      rainfallMm: rain,
      condition: condition,
      city: city,
    );
  }
}
