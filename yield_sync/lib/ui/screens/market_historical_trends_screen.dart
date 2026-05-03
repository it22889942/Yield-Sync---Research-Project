import 'dart:math';
import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../../utils/app_colors.dart';
import '../widgets/app_shell.dart';
import '../../services/market_api.dart';
import '../../services/market_cache_service.dart';

class MarketHistoricalTrendsScreen extends StatefulWidget {
  final String initialCrop;
  const MarketHistoricalTrendsScreen({super.key, this.initialCrop = "All"});

  @override
  State<MarketHistoricalTrendsScreen> createState() =>
      _MarketHistoricalTrendsScreenState();
}

class _MarketHistoricalTrendsScreenState
    extends State<MarketHistoricalTrendsScreen> {
  late String _crop;
  String _market = "All";

  bool _loading = false;
  String? _error;

  Map<String, dynamic>? _stats;
  List<Map<String, dynamic>> _series = [];
  List<Map<String, dynamic>> _recent = [];

  List<String> _crops = const [
    "All",
    "Rice",
    "Radish",
    "Red Onion",
    "Beetroot"
  ];
  List<String> _markets = const ["All"]; // will load based on crop

  // ✅ UX: chart interactions
  int? _touchedLineIdx;
  FlSpot? _touchedSpot;

  @override
  void initState() {
    super.initState();
    _crop = widget.initialCrop;
    _loadCacheThenFetch();
  }

  Future<void> _loadCacheThenFetch() async {
    await _loadMarkets();

    // ✅ 1) Load cache first
    try {
      final cached =
          await MarketCacheService.loadTrends(crop: _crop, market: _market);
      if (cached != null) {
        _applyTrendsPayload(cached);
      }
    } catch (_) {}

    // ✅ 2) Fetch fresh
    _fetch();
  }

  Future<void> _loadMarkets() async {
    try {
      if (_crop == "All") {
        if (mounted) setState(() => _markets = const ["All"]);
        return;
      }
      final markets = await MarketApi.getMarkets(_crop);
      if (!mounted) return;
      setState(() => _markets = ["All", ...markets]);
      if (!_markets.contains(_market)) setState(() => _market = "All");
    } catch (_) {
      if (mounted) setState(() => _markets = const ["All"]);
    }
  }

  void _applyTrendsPayload(Map<String, dynamic> res) {
    final stats =
        (res["stats"] as Map?)?.map((k, v) => MapEntry(k.toString(), v));
    final series = (res["series"] as List? ?? [])
        .map((e) => (e as Map).map((k, v) => MapEntry(k.toString(), v)))
        .toList();
    final recent = (res["recent"] as List? ?? [])
        .map((e) => (e as Map).map((k, v) => MapEntry(k.toString(), v)))
        .toList();

    if (!mounted) return;
    setState(() {
      _stats = stats?.cast<String, dynamic>();
      _series = series.cast<Map<String, dynamic>>();
      _recent = recent.cast<Map<String, dynamic>>();
    });
  }

  Future<void> _fetch() async {
    setState(() {
      _loading = true;
      _error = null;
      _touchedLineIdx = null;
      _touchedSpot = null;
    });

    try {
      final res = await MarketApi.trends(
        crop: _crop,
        market: _market,
        maxPoints: 420,
        recent: 50,
      );

      // ✅ Save to Firestore
      try {
        await MarketCacheService.saveTrends(
          crop: _crop,
          market: _market,
          payload: res,
        );
      } catch (_) {}

      _applyTrendsPayload(res);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _refresh() async {
    await _fetch();
  }

  @override
  Widget build(BuildContext context) {
    return AppShell(
      currentIndex: 0,
      child: Column(
        children: [
          // ===== HEADER (modern + chips + refresh) =====
          Container(
            width: double.infinity,
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 18),
            decoration: const BoxDecoration(
              gradient: AppColors.heroGradient,
              borderRadius: BorderRadius.only(
                bottomLeft: Radius.circular(28),
                bottomRight: Radius.circular(28),
              ),
            ),
            child: Stack(
              children: [
                const Positioned(
                  right: -16,
                  top: 10,
                  child: Opacity(
                    opacity: 0.10,
                    child: Icon(Icons.timeline_rounded,
                        size: 170, color: Colors.white),
                  ),
                ),
                Column(
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
                          "Historical Trends",
                          style: TextStyle(
                            color: Colors.white.withOpacity(0.92),
                            fontWeight: FontWeight.w900,
                            fontSize: 16,
                          ),
                        ),
                        const Spacer(),
                        InkWell(
                          borderRadius: BorderRadius.circular(999),
                          onTap: _loading ? null : _fetch,
                          child: CircleAvatar(
                            radius: 18,
                            backgroundColor: Colors.white.withOpacity(0.12),
                            child: _loading
                                ? const SizedBox(
                                    width: 16,
                                    height: 16,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                      color: Colors.white,
                                    ),
                                  )
                                : const Icon(Icons.refresh_rounded,
                                    color: Colors.white),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 10),
                    Text(
                      "Track prices over time",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.98),
                        fontWeight: FontWeight.w900,
                        fontSize: 22,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      "Tap the chart to inspect values • Pull to refresh",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.72),
                        fontWeight: FontWeight.w600,
                        fontSize: 12.8,
                      ),
                    ),
                    const SizedBox(height: 12),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        _HeaderChip(icon: Icons.eco_rounded, text: _crop),
                        _HeaderChip(
                            icon: Icons.storefront_rounded, text: _market),
                        _HeaderChip(
                          icon: Icons.data_usage_rounded,
                          text: "${(_stats?["total_records"] ?? "—")} records",
                        ),
                      ],
                    ),
                  ],
                ),
              ],
            ),
          ),

          const SizedBox(height: 12),

          // ===== BODY =====
          Expanded(
            child: Container(
              width: double.infinity,
              margin: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: AppColors.border),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.05),
                    blurRadius: 18,
                    offset: const Offset(0, 12),
                  ),
                ],
              ),
              child: RefreshIndicator(
                onRefresh: _refresh,
                color: AppColors.darkGreen,
                child: ListView(
                  physics: const BouncingScrollPhysics(
                    parent: AlwaysScrollableScrollPhysics(),
                  ),
                  padding: const EdgeInsets.fromLTRB(16, 16, 16, 16),
                  children: [
                    // ✅ Stats row (kept) + nicer spacing
                    _StatsRow(stats: _stats),
                    const SizedBox(height: 14),

                    // ✅ Filters
                    Row(
                      children: [
                        Expanded(
                          child: _DropdownBox(
                            label: "Crop",
                            value: _crop,
                            items: _crops,
                            onChanged: (v) async {
                              setState(() => _crop = v);
                              await _loadMarkets();
                              await _loadCacheThenFetch();
                            },
                          ),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: _DropdownBox(
                            label: "Market",
                            value: _market,
                            items: _markets,
                            onChanged: (v) async {
                              setState(() => _market = v);
                              await _loadCacheThenFetch();
                            },
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 14),

                    // ✅ Loading / error banners
                    if (_loading)
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: AppColors.surface,
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: AppColors.border),
                        ),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            const SizedBox(
                              height: 18,
                              width: 18,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            ),
                            const SizedBox(width: 10),
                            Text(
                              "Loading trends...",
                              style: TextStyle(
                                fontWeight: FontWeight.w800,
                                color: AppColors.textDark.withOpacity(0.80),
                              ),
                            ),
                          ],
                        ),
                      ),

                    if (_error != null && !_loading)
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.red.withOpacity(0.06),
                          borderRadius: BorderRadius.circular(16),
                          border:
                              Border.all(color: Colors.red.withOpacity(0.2)),
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.error_outline_rounded,
                                color: Colors.red),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                _error!,
                                style: const TextStyle(
                                  color: Colors.red,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),

                    const SizedBox(height: 10),

                    // ✅ Chart + Inspector (new)
                    _TrendChartCard(
                      series: _series,
                      touchedLineIdx: _touchedLineIdx,
                      touchedSpot: _touchedSpot,
                      onTouch: (lineIdx, spot) {
                        setState(() {
                          _touchedLineIdx = lineIdx;
                          _touchedSpot = spot;
                        });
                      },
                      onClearTouch: () {
                        setState(() {
                          _touchedLineIdx = null;
                          _touchedSpot = null;
                        });
                      },
                    ),

                    const SizedBox(height: 14),

                    // ✅ Recent list
                    _RecentEntriesCard(rows: _recent),
                  ],
                ),
              ),
            ),
          )
        ],
      ),
    );
  }
}

// ===================== small UI helpers =====================
class _HeaderChip extends StatelessWidget {
  final IconData icon;
  final String text;
  const _HeaderChip({required this.icon, required this.text});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.10),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: Colors.white.withOpacity(0.14)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 16, color: Colors.white.withOpacity(0.95)),
          const SizedBox(width: 8),
          Text(
            text,
            style: TextStyle(
              color: Colors.white.withOpacity(0.92),
              fontWeight: FontWeight.w800,
              fontSize: 12.2,
            ),
          ),
        ],
      ),
    );
  }
}

// ---------- Widgets below (UPGRADED but keep your logic) ----------
class _StatsRow extends StatelessWidget {
  final Map<String, dynamic>? stats;
  const _StatsRow({required this.stats});
  String _fmt(dynamic v) => (v ?? "").toString();

  @override
  Widget build(BuildContext context) {
    final s = stats;
    return Row(
      children: [
        Expanded(
          child: _StatCard(
              title: "Total Records", value: _fmt(s?["total_records"])),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: _StatCard(title: "Days", value: _fmt(s?["days_of_data"])),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: _StatCard(title: "First", value: _fmt(s?["first_date"])),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: _StatCard(title: "Last", value: _fmt(s?["last_date"])),
        ),
      ],
    );
  }
}

class _StatCard extends StatelessWidget {
  final String title;
  final String value;
  const _StatCard({required this.title, required this.value});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 12, 12, 12),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: TextStyle(
              fontWeight: FontWeight.w900,
              fontSize: 11.5,
              color: AppColors.textDark.withOpacity(0.65),
            ),
          ),
          const SizedBox(height: 8),
          Text(
            value.isEmpty ? "—" : value,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              fontWeight: FontWeight.w900,
              fontSize: 18,
              color: AppColors.textDark,
            ),
          ),
        ],
      ),
    );
  }
}

class _DropdownBox extends StatelessWidget {
  final String label;
  final String value;
  final List<String> items;
  final ValueChanged<String> onChanged;

  const _DropdownBox({
    required this.label,
    required this.value,
    required this.items,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.fromLTRB(12, 10, 12, 10),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: TextStyle(
              fontWeight: FontWeight.w900,
              fontSize: 11.5,
              color: AppColors.textDark.withOpacity(0.6),
            ),
          ),
          DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: value,
              isExpanded: true,
              items: items
                  .map((x) => DropdownMenuItem(
                        value: x,
                        child: Text(
                          x,
                          style: const TextStyle(fontWeight: FontWeight.w800),
                        ),
                      ))
                  .toList(),
              onChanged: (v) => onChanged(v ?? value),
            ),
          ),
        ],
      ),
    );
  }
}

class _TrendChartCard extends StatelessWidget {
  final List<Map<String, dynamic>> series;

  // ✅ UX: inspector
  final int? touchedLineIdx;
  final FlSpot? touchedSpot;
  final void Function(int lineIdx, FlSpot spot) onTouch;
  final VoidCallback onClearTouch;

  const _TrendChartCard({
    required this.series,
    required this.touchedLineIdx,
    required this.touchedSpot,
    required this.onTouch,
    required this.onClearTouch,
  });

  @override
  Widget build(BuildContext context) {
    final inspectorVisible = touchedLineIdx != null && touchedSpot != null;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(14, 14, 14, 14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.04),
            blurRadius: 14,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 38,
                height: 38,
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.16),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppColors.border),
                ),
                child: const Icon(Icons.show_chart_rounded,
                    color: AppColors.darkGreen),
              ),
              const SizedBox(width: 10),
              const Expanded(
                child: Text(
                  "Price Trends",
                  style: TextStyle(
                    color: AppColors.textDark,
                    fontWeight: FontWeight.w900,
                    fontSize: 14.5,
                  ),
                ),
              ),
              if (inspectorVisible)
                TextButton.icon(
                  onPressed: onClearTouch,
                  icon: const Icon(Icons.close_rounded, size: 18),
                  label: const Text(
                    "Clear",
                    style: TextStyle(fontWeight: FontWeight.w900),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            "Tap line for exact value",
            style: TextStyle(
              color: AppColors.textDark.withOpacity(0.60),
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 12),

          // ✅ Inspector card
          if (inspectorVisible) ...[
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: AppColors.surface,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppColors.border),
              ),
              child: Row(
                children: [
                  const Icon(Icons.my_location_rounded,
                      size: 18, color: AppColors.darkGreen),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      "Point: x=${touchedSpot!.x.toInt()} • y=${touchedSpot!.y.toStringAsFixed(0)}",
                      style: const TextStyle(
                        fontWeight: FontWeight.w900,
                        color: AppColors.textDark,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 10),
          ],

          SizedBox(
            height: 270,
            child: series.isEmpty
                ? Center(
                    child: Text(
                      "No trend data",
                      style: TextStyle(
                        color: AppColors.textDark.withOpacity(0.6),
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  )
                : LineChart(_buildChartData(series)),
          ),
        ],
      ),
    );
  }

  LineChartData _buildChartData(List<Map<String, dynamic>> s) {
    final allCounts =
        s.map((e) => (e["prices"] as List?)?.length ?? 0).toList();
    final maxLen = allCounts.isEmpty ? 0 : allCounts.reduce(max);

    double globalMaxY = 0;
    for (final one in s) {
      final prices = (one["prices"] as List? ?? []);
      for (final p in prices) {
        final v =
            (p is num) ? p.toDouble() : double.tryParse(p.toString()) ?? 0;
        if (v > globalMaxY) globalMaxY = v;
      }
    }
    if (globalMaxY <= 0) globalMaxY = 1;

    final lines = <LineChartBarData>[];
    for (int i = 0; i < s.length; i++) {
      final prices = (s[i]["prices"] as List? ?? []);
      final spots = <FlSpot>[];
      for (int x = 0; x < prices.length; x++) {
        final v = (prices[x] is num)
            ? (prices[x] as num).toDouble()
            : double.tryParse(prices[x].toString()) ?? 0;
        spots.add(FlSpot(x.toDouble(), v));
      }

      final colors = [
        Colors.blue,
        Colors.red,
        Colors.green,
        Colors.purple,
        Colors.orange,
        Colors.teal,
      ];
      final c = colors[i % colors.length];

      lines.add(
        LineChartBarData(
          spots: spots,
          isCurved: true, // ✅ smoother modern look
          barWidth: 2.4,
          color: c,
          dotData: const FlDotData(show: false),
        ),
      );
    }

    return LineChartData(
      minX: 0,
      maxX: max(0, maxLen - 1).toDouble(),
      minY: 0,
      maxY: globalMaxY * 1.05,
      gridData: FlGridData(show: true),
      borderData: FlBorderData(show: true),

      // ✅ touch + tooltip
      lineTouchData: LineTouchData(
        handleBuiltInTouches: true,
        touchTooltipData: LineTouchTooltipData(
          tooltipRoundedRadius: 12,
          tooltipPadding: const EdgeInsets.all(10),
          getTooltipItems: (touchedSpots) {
            return touchedSpots.map((ts) {
              return LineTooltipItem(
                "x=${ts.x.toInt()}  y=${ts.y.toStringAsFixed(0)}",
                const TextStyle(
                  fontWeight: FontWeight.w900,
                  color: AppColors.textDark,
                  fontSize: 12,
                ),
              );
            }).toList();
          },
        ),
        touchCallback: (event, resp) {
          if (resp == null || resp.lineBarSpots == null) return;
          if (event.isInterestedForInteractions &&
              resp.lineBarSpots!.isNotEmpty) {
            final s0 = resp.lineBarSpots!.first;
            onTouch(s0.barIndex, s0);
          }
        },
      ),

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
            interval: max(50, (globalMaxY / 5).floorToDouble()),
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

class _RecentEntriesCard extends StatelessWidget {
  final List<Map<String, dynamic>> rows;
  const _RecentEntriesCard({required this.rows});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(14, 14, 14, 14),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            "Recent Entries (Last 50)",
            style: TextStyle(
              fontWeight: FontWeight.w900,
              fontSize: 14,
              color: AppColors.textDark,
            ),
          ),
          const SizedBox(height: 10),
          SizedBox(
            height: 320,
            child: rows.isEmpty
                ? Center(
                    child: Text(
                      "No recent entries",
                      style: TextStyle(
                        color: AppColors.textDark.withOpacity(0.6),
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                  )
                : ListView.separated(
                    physics: const BouncingScrollPhysics(),
                    itemCount: rows.length,
                    separatorBuilder: (_, __) => Divider(
                      height: 14,
                      color: AppColors.border.withOpacity(0.9),
                    ),
                    itemBuilder: (_, i) {
                      final r = rows[i];
                      final date = (r["date"] ?? "").toString();
                      final market = (r["market"] ?? "").toString();
                      final item = (r["item"] ?? "").toString();
                      final price = (r["price"] ?? 0).toString();

                      return Row(
                        children: [
                          Expanded(
                            flex: 4,
                            child: Text(
                              date,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                fontWeight: FontWeight.w800,
                                color: AppColors.textDark,
                                fontSize: 12,
                              ),
                            ),
                          ),
                          Expanded(
                            flex: 3,
                            child: Text(
                              market,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(
                                fontWeight: FontWeight.w800,
                                color: AppColors.textDark.withOpacity(0.75),
                                fontSize: 12,
                              ),
                            ),
                          ),
                          Expanded(
                            flex: 3,
                            child: Text(
                              item,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: TextStyle(
                                fontWeight: FontWeight.w900,
                                color: AppColors.textDark.withOpacity(0.85),
                                fontSize: 12,
                              ),
                            ),
                          ),
                          Expanded(
                            flex: 2,
                            child: Text(
                              price,
                              textAlign: TextAlign.right,
                              style: const TextStyle(
                                fontWeight: FontWeight.w900,
                                color: AppColors.darkGreen,
                                fontSize: 12,
                              ),
                            ),
                          ),
                        ],
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }
}
