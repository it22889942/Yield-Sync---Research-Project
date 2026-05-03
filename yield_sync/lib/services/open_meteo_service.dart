import 'dart:convert';
import 'package:http/http.dart' as http;

class OpenMeteoWeather {
  final double temperatureC;
  final double windSpeed;
  final int weatherCode;

  OpenMeteoWeather({
    required this.temperatureC,
    required this.windSpeed,
    required this.weatherCode,
  });

  String get weatherLabel => _codeToLabel(weatherCode);

  static String _codeToLabel(int code) {
    // Simple mapping (you can extend)
    if (code == 0) return "Clear";
    if (code == 1 || code == 2) return "Partly cloudy";
    if (code == 3) return "Cloudy";
    if (code == 45 || code == 48) return "Fog";
    if (code >= 51 && code <= 67) return "Drizzle";
    if (code >= 71 && code <= 77) return "Snow";
    if (code >= 80 && code <= 82) return "Rain showers";
    if (code >= 95) return "Thunderstorm";
    return "Unknown";
  }
}

class OpenMeteoService {
  static Future<OpenMeteoWeather> current({
    required double lat,
    required double lon,
  }) async {
    final uri = Uri.parse(
      "https://api.open-meteo.com/v1/forecast"
      "?latitude=$lat&longitude=$lon"
      "&current_weather=true"
      "&timezone=auto",
    );

    final res = await http.get(uri);
    if (res.statusCode != 200) {
      throw Exception("Open-Meteo error ${res.statusCode}: ${res.body}");
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    final cw = (data["current_weather"] ?? {}) as Map<String, dynamic>;

    return OpenMeteoWeather(
      temperatureC: (cw["temperature"] as num?)?.toDouble() ?? 0.0,
      windSpeed: (cw["windspeed"] as num?)?.toDouble() ?? 0.0,
      weatherCode: (cw["weathercode"] as num?)?.toInt() ?? -1,
    );
  }
}
