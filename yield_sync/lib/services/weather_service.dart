import 'dart:convert';
import 'dart:async';
import 'package:http/http.dart' as http;

class WeatherInfo {
  final double tempC;
  final double humidity;
  final double rainfallMm;
  final String condition;
  final String city;
  /// Wind speed in km/h (from API or mock).
  final double windSpeedKmh;

  WeatherInfo({
    required this.tempC,
    required this.humidity,
    required this.rainfallMm,
    required this.condition,
    required this.city,
    this.windSpeedKmh = 0,
  });
}

class WeatherService {
  // ✅ If you keep this as placeholder -> it will AUTO USE MOCK
  static const String openWeatherKey = "YOUR_OPENWEATHER_API_KEY";

  static Future<WeatherInfo> fetchByLatLon(double lat, double lon) async {
    // ✅ UI STAGE (MOCK)
    if (openWeatherKey == "YOUR_OPENWEATHER_API_KEY") {
      await Future.delayed(const Duration(milliseconds: 600));
      // Demo data: tied to time so the UI visibly changes between sessions.
      final now = DateTime.now();
      final seed = now.day + now.hour + now.minute;
      final variants = <({double t, int h, double rain, String c, double w})>[
        (t: 27.2, h: 74, rain: 8.2, c: "Cloudy", w: 12),
        (t: 29.1, h: 68, rain: 0.0, c: "Clear", w: 8),
        (t: 26.4, h: 81, rain: 22.0, c: "Rain", w: 18),
        (t: 28.0, h: 71, rain: 3.5, c: "Partly cloudy", w: 15),
      ];
      final v = variants[seed % variants.length];
      return WeatherInfo(
        tempC: v.t,
        humidity: v.h.toDouble(),
        rainfallMm: v.rain,
        condition: v.c,
        city: "Malabe, Colombo",
        windSpeedKmh: v.w,
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

    double windMs = 0;
    final windObj = data["wind"];
    if (windObj is Map && windObj["speed"] != null) {
      windMs = (windObj["speed"] as num).toDouble();
    }
    final windKmh = windMs * 3.6;

    return WeatherInfo(
      tempC: temp,
      humidity: humidity,
      rainfallMm: rain,
      condition: condition,
      city: city,
      windSpeedKmh: windKmh,
    );
  }
}
