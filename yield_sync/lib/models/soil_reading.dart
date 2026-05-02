import 'package:cloud_firestore/cloud_firestore.dart';

class SoilReading {
  final double temperature;
  final double humidity;
  final double ph;
  final double n;
  final double p;
  final double k;
  final double moisture;
  final double ec;
  final DateTime? updatedAt;

  SoilReading({
    required this.temperature,
    required this.humidity,
    required this.ph,
    required this.n,
    required this.p,
    required this.k,
    required this.moisture,
    required this.ec,
    required this.updatedAt,
  });

  static double _toDouble(dynamic v) {
    if (v is num) return v.toDouble();
    return double.tryParse(v?.toString() ?? "") ?? 0.0;
  }

  factory SoilReading.fromDoc(DocumentSnapshot<Map<String, dynamic>> doc) {
    final d = doc.data() ?? {};
    final ts = d["updatedAt"];
    DateTime? dt;
    if (ts is Timestamp) dt = ts.toDate();

    return SoilReading(
      temperature: _toDouble(d["temperature"]),
      humidity: _toDouble(d["humidity"]),
      ph: _toDouble(d["ph"]),
      n: _toDouble(d["N"]),
      p: _toDouble(d["P"]),
      k: _toDouble(d["K"]),
      moisture: _toDouble(d["moisture"]), // if missing => 0
      ec: _toDouble(d["ec"]),
      updatedAt: dt,
    );
  }
}
