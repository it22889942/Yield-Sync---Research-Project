// lib/services/fertiliser_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import '../config/api_client.dart';

class FertiliserPredictResult {
  final String fertiliserType;
  final double? yieldKgPerAcre;
  final Map<String, dynamic> raw;

  FertiliserPredictResult({
    required this.fertiliserType,
    required this.yieldKgPerAcre,
    required this.raw,
  });

  factory FertiliserPredictResult.fromBackend(Map<String, dynamic> json) {
    final ft = (json["fertiliser_type"] ?? "").toString();

    final y = json["yield_kg_per_acre"];
    double? ypa;
    if (y is num) {
      ypa = y.toDouble();
    } else if (y is String) {
      ypa = double.tryParse(y);
    } else {
      ypa = null;
    }

    return FertiliserPredictResult(
      fertiliserType: ft,
      yieldKgPerAcre: ypa,
      raw: json,
    );
  }

  Map<String, dynamic> toJson() => {
        "fertiliserType": fertiliserType,
        "yieldKgPerAcre": yieldKgPerAcre,
        "raw": raw,
      };
}

class TotalYieldResult {
  final double areaAcres;
  final double yieldKgPerAcre;
  final double totalYieldKg;
  final String source;

  TotalYieldResult({
    required this.areaAcres,
    required this.yieldKgPerAcre,
    required this.totalYieldKg,
    required this.source,
  });

  factory TotalYieldResult.fromBackend(Map<String, dynamic> json) {
    double _toDouble(dynamic v) {
      if (v is num) return v.toDouble();
      return double.tryParse(v?.toString() ?? "") ?? 0.0;
    }

    return TotalYieldResult(
      areaAcres: _toDouble(json["area_acres"]),
      yieldKgPerAcre: _toDouble(json["yield_kg_per_acre"]),
      totalYieldKg: _toDouble(json["total_yield_kg"]),
      source: (json["source"] ?? "").toString(),
    );
  }
}

class FertiliserService {
  static Future<FertiliserPredictResult> predict({
    required double temperature,
    required double ph,
    required double nitrogen,
    required double phosphorous,
    required double potassium,
    required String crop,
    required String growthStage,
  }) async {
    final ts = DateTime.now().millisecondsSinceEpoch;
    final uri = ApiClient.url("/api/fertiliser/predict?ts=$ts");

    final body = {
      "temperature": temperature,
      "ph": ph,
      "nitrogen": nitrogen,
      "phosphorous": phosphorous,
      "potassium": potassium,
      "crop": crop, // ✅ EXACT
      "growth_stage": growthStage, // ✅ lowercase
    };

    final res = await http.post(
      uri,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
      },
      body: jsonEncode(body),
    );

    if (res.statusCode != 200) {
      throw Exception("Backend ${res.statusCode}: ${res.body}");
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    return FertiliserPredictResult.fromBackend(data);
  }

  static Future<TotalYieldResult> totalYield({
    required double areaAcres,
    required double yieldKgPerAcre,
  }) async {
    final ts = DateTime.now().millisecondsSinceEpoch;
    final uri = ApiClient.url("/api/fertiliser/total-yield?ts=$ts");

    final body = {
      "area_acres": areaAcres,
      "yield_kg_per_acre": yieldKgPerAcre,
    };

    final res = await http.post(
      uri,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
      },
      body: jsonEncode(body),
    );

    if (res.statusCode != 200) {
      throw Exception("Backend ${res.statusCode}: ${res.body}");
    }

    final data = jsonDecode(res.body) as Map<String, dynamic>;
    return TotalYieldResult.fromBackend(data);
  }
}
