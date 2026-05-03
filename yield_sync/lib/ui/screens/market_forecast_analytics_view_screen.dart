import 'dart:math';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';

import '../../utils/app_colors.dart';
import '../widgets/app_shell.dart';
import '../../services/market_api.dart';

class MarketForecastAnalyticsViewScreen extends StatefulWidget {
  const MarketForecastAnalyticsViewScreen({
    super.key,
    this.crop,
    this.market,
    this.quantityKg,
    this.daysSinceHarvest,
  });

  final String? crop;
  final String? market;
  final int? quantityKg;
  final int? daysSinceHarvest;

  @override
  State<MarketForecastAnalyticsViewScreen> createState() =>
      _MarketForecastAnalyticsViewScreenState();
}

class _MarketForecastAnalyticsViewScreenState
    extends State<MarketForecastAnalyticsViewScreen> {
  bool _loading = false;
  String? _error;

  // backend response parts
  Map<String, dynamic>? _stats;
  List<Map<String, dynamic>> _series = [];

  @override
  void initState() {
    super.initState();
    _fetch();
  }

  Future<void> _fetch() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final res = await MarketApi.forecastAnalytics(
        crop: widget.crop ?? "All",
        market: widget.market ?? "All",
        maxPoints: 260,
      );

      final stats =
          (res["stats"] as Map?)?.map((k, v) => MapEntry(k.toString(), v));
      final series = (res["series"] as List? ?? [])
          .map((e) => (e as Map).map((k, v) => MapEntry(k.toString(), v)))
          .toList();

      setState(() {
        _stats = stats?.cast<String, dynamic>();
        _series = series.cast<Map<String, dynamic>>();
      });
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cropLabel = widget.crop ?? "All crops";
    final marketLabel = widget.market ?? "All markets";

    return AppShell(
      currentIndex: 0,
      child: Column(
        children: [
          // ===== HEADER =====
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 18),
            decoration: const BoxDecoration(
              gradient: AppColors.heroGradient,
              borderRadius: BorderRadius.only(
                bottomLeft: Radius.circular(26),
                bottomRight: Radius.circular(26),
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    IconButton(
                      onPressed: () => Navigator.pop(context),
                      icon: const Icon(Icons.arrow_back_ios_new_rounded),
                      color: Colors.white,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      "Price & Demand Forecast",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.92),
                        fontWeight: FontWeight.w900,
                        fontSize: 16,
                      ),
                    ),
                    const Spacer(),
                    CircleAvatar(
                      radius: 18,
                      backgroundColor: Colors.white.withOpacity(0.12),
                      child:
                          const Icon(Icons.person_rounded, color: Colors.white),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Text(
                  "Forecast and Analytics",
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.96),
                    fontWeight: FontWeight.w900,
                    fontSize: 22,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  "Market trend analytics from backend",
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.72),
                    fontWeight: FontWeight.w600,
                    fontSize: 12.8,
                  ),
                ),
                const SizedBox(height: 10),

                // chip showing selected filters (same as you had)
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.10),
                    borderRadius: BorderRadius.circular(999),
                    border: Border.all(color: Colors.white.withOpacity(0.14)),
                  ),
                  child: Text(
                    "$cropLabel • $marketLabel",
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.90),
                      fontWeight: FontWeight.w800,
                      fontSize: 12.2,
                    ),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 12),

          // ===== BODY =====
          Expanded(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 18),
              children: [
                // Chart card + legend card (same structure)
                Container(
                  padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(22),
                    border: Border.all(color: AppColors.border),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.06),
                        blurRadius: 22,
                        offset: const Offset(0, 14),
                      ),
                    ],
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      // Chart area
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(18),
                          border: Border.all(color: AppColors.border),
                        ),
                        child: AspectRatio(
                          aspectRatio: 16 / 10,
                          child: _loading
                              ? const Center(
                                  child: SizedBox(
                                    height: 22,
                                    width: 22,
                                    child: CircularProgressIndicator(
                                        strokeWidth: 2),
                                  ),
                                )
                              : (_error != null)
                                  ? Center(
                                      child: Text(
                                        _error!,
                                        textAlign: TextAlign.center,
                                        style: const TextStyle(
                                          color: Colors.red,
                                          fontWeight: FontWeight.w800,
                                        ),
                                      ),
                                    )
                                  : (_series.isEmpty)
                                      ? Center(
                                          child: Text(
                                            "No data",
                                            style: TextStyle(
                                              color: AppColors.textDark
                                                  .withOpacity(0.6),
                                              fontWeight: FontWeight.w800,
                                            ),
                                          ),
                                        )
                                      : LineChart(_buildLineChartData(_series)),
                        ),
                      ),

                      const SizedBox(height: 14),

                      // Legend card (auto from backend series)
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.fromLTRB(14, 14, 14, 14),
                        decoration: BoxDecoration(
                          color: AppColors.primary.withOpacity(0.18),
                          borderRadius: BorderRadius.circular(18),
                          border: Border.all(color: AppColors.border),
                        ),
                        child: _series.isEmpty
                            ? Text(
                                "Legend will show here",
                                style: TextStyle(
                                  color: AppColors.textDark.withOpacity(0.6),
                                  fontWeight: FontWeight.w800,
                                ),
                              )
                            : Column(
                                children: _series
                                    .take(6)
                                    .toList()
                                    .asMap()
                                    .entries
                                    .map((e) {
                                  final i = e.key;
                                  final label =
                                      (e.value["label"] ?? "Series").toString();
                                  final color =
                                      _lineColors[i % _lineColors.length];
                                  return Padding(
                                    padding: EdgeInsets.only(
                                        bottom:
                                            i == _series.length - 1 ? 0 : 10),
                                    child:
                                        _LegendRow(color: color, label: label),
                                  );
                                }).toList(),
                              ),
                      ),

                      const SizedBox(height: 12),

                      // Optional small stats line (doesn't change UI much)
                      if (_stats != null)
                        Text(
                          "Records: ${_stats!["total_records"] ?? "-"}  •  Range: ${_stats!["first_date"] ?? "-"} → ${_stats!["last_date"] ?? "-"}",
                          style: TextStyle(
                            color: AppColors.textDark.withOpacity(0.55),
                            fontWeight: FontWeight.w800,
                            fontSize: 12.2,
                          ),
                        ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ---------- chart helpers ----------
  static const _lineColors = <Color>[
    Color(0xFF8B5CF6), // purple
    Color(0xFFEF4444), // red
    Color(0xFF22C55E), // green
    Color(0xFF06B6D4), // cyan
    Color(0xFFF59E0B), // amber
    Color(0xFF3B82F6), // blue
  ];

  LineChartData _buildLineChartData(List<Map<String, dynamic>> series) {
    // compute maxLen and maxY
    int maxLen = 0;
    double maxY = 0;

    for (final s in series) {
      final prices = (s["prices"] as List? ?? []);
      maxLen = max(maxLen, prices.length);
      for (final p in prices) {
        final v =
            (p is num) ? p.toDouble() : double.tryParse(p.toString()) ?? 0;
        maxY = max(maxY, v);
      }
    }
    if (maxY <= 0) maxY = 1;

    final lines = <LineChartBarData>[];
    for (int i = 0; i < series.length; i++) {
      final prices = (series[i]["prices"] as List? ?? []);
      final spots = <FlSpot>[];
      for (int x = 0; x < prices.length; x++) {
        final y = (prices[x] is num)
            ? (prices[x] as num).toDouble()
            : double.tryParse(prices[x].toString()) ?? 0;
        spots.add(FlSpot(x.toDouble(), y));
      }

      lines.add(
        LineChartBarData(
          spots: spots,
          isCurved: false,
          barWidth: 2.0,
          color: _lineColors[i % _lineColors.length],
          dotData: const FlDotData(show: false),
        ),
      );
    }

    return LineChartData(
      minX: 0,
      maxX: max(0, maxLen - 1).toDouble(),
      minY: 0,
      maxY: maxY * 1.05,
      gridData: FlGridData(show: true),
      borderData: FlBorderData(show: true),
      titlesData: FlTitlesData(
        rightTitles:
            const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
        bottomTitles: AxisTitles(
          sideTitles: SideTitles(
            showTitles: true,
            interval: max(1, (maxLen / 4).floor()).toDouble(),
            getTitlesWidget: (v, meta) => Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(
                v.toInt().toString(),
                style: TextStyle(
                  fontSize: 10,
                  color: AppColors.textDark.withOpacity(0.55),
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
          ),
        ),
        leftTitles: AxisTitles(
          sideTitles: SideTitles(
            showTitles: true,
            reservedSize: 36,
            interval: max(50, (maxY / 5).floorToDouble()),
            getTitlesWidget: (v, meta) => Text(
              v.toInt().toString(),
              style: TextStyle(
                fontSize: 10,
                color: AppColors.textDark.withOpacity(0.55),
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
        ),
      ),
      lineBarsData: lines,
    );
  }
}

class _LegendRow extends StatelessWidget {
  final Color color;
  final String label;

  const _LegendRow({required this.color, required this.label});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 10),
        Text(
          label,
          style: TextStyle(
            color: AppColors.textDark.withOpacity(0.85),
            fontWeight: FontWeight.w800,
          ),
        ),
      ],
    );
  }
}
