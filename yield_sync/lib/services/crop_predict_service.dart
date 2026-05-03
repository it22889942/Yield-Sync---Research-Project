import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config/api_client.dart';

class CropPredictResult {
  final String prediction;
  final List<Map<String, dynamic>> probs; // [{class, prob}]
  final Map<String, dynamic> raw; // full backend response

  CropPredictResult({
    required this.prediction,
    required this.probs,
    required this.raw,
  });

  factory CropPredictResult.fromBackend(Map<String, dynamic> json) {
    // backend => { ok, results:[{prediction, probs:[]}] }
    final results = (json["results"] is List) ? List.from(json["results"]) : [];
    final first = results.isNotEmpty && results.first is Map<String, dynamic>
        ? results.first as Map<String, dynamic>
        : <String, dynamic>{};

    final prediction = (first["prediction"] ?? "").toString();
    final probs = (first["probs"] is List)
        ? List<Map<String, dynamic>>.from(first["probs"])
        : <Map<String, dynamic>>[];

    return CropPredictResult(
      prediction: prediction,
      probs: probs,
      raw: json,
    );
  }

  Map<String, dynamic> toJson() => {
        "prediction": prediction,
        "probs": probs,
        "raw": raw,
      };
}

class CropPredictService {
  static Future<CropPredictResult> predict({
    required double n,
    required double p,
    required double k,
    required double ph,
    required double temp,
    required double humidity,
    required double lat,
    required double lon,
  }) async {
    final uri = ApiClient.url("/api/crop/predict");

    final body = {
      "N": n,
      "P": p,
      "K": k,
      "ph": ph,
      "temperature": temp,
      "humidity": humidity,
      // ✅ send location so backend auto-fills Rainfall
      "lat": lat,
      "lon": lon,
    };

    final res = await http.post(
      uri,
      headers: {"Content-Type": "application/json"},
      body: jsonEncode(body),
    );

    if (res.statusCode != 200) {
      throw Exception("Backend ${res.statusCode}: ${res.body}");
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    return CropPredictResult.fromBackend(data);
  }
}
