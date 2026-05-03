import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../widgets/app_shell.dart';
import '../../services/market_api.dart';
import '../../services/market_cache_service.dart';
import 'market_historical_trends_screen.dart';

class MarketComparisonHistoryScreen extends StatefulWidget {
  const MarketComparisonHistoryScreen({super.key});

  @override
  State<MarketComparisonHistoryScreen> createState() =>
      _MarketComparisonHistoryScreenState();
}

class _MarketComparisonHistoryScreenState
    extends State<MarketComparisonHistoryScreen> {
  String _crop = "Rice";
  int _days = 7;

  bool _loading = false;
  String? _error;

  List<Map<String, dynamic>> _rows = [];
  List<Map<String, dynamic>> _top = [];

  @override
  void initState() {
    super.initState();
    _loadCacheThenFetch();
  }

  Future<void> _loadCacheThenFetch() async {
    // ✅ 1) Load cached snapshot (fast)
    try {
      final cached =
          await MarketCacheService.loadAvgByMarket(crop: _crop, days: _days);

      if (cached != null) {
        final rows = (cached["rows"] as List? ?? [])
            .map((e) => (e as Map).map((k, v) => MapEntry(k.toString(), v)))
            .toList();

        final top = (cached["top"] as List? ?? [])
            .map((e) => (e as Map).map((k, v) => MapEntry(k.toString(), v)))
            .toList();

        if (mounted) {
          setState(() {
            _rows = rows.cast<Map<String, dynamic>>();
            _top = top.cast<Map<String, dynamic>>();
          });
        }
      }
    } catch (_) {
      // ignore cache errors
    }

    // ✅ 2) Fetch fresh data from backend
    _fetch();
  }

  Future<void> _fetch() async {
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final res = await MarketApi.avgByMarket(crop: _crop, days: _days);

      final rows = (res["rows"] as List? ?? [])
          .map((e) => (e as Map).map((k, v) => MapEntry(k.toString(), v)))
          .toList();

      final top = (res["top"] as List? ?? [])
          .map((e) => (e as Map).map((k, v) => MapEntry(k.toString(), v)))
          .toList();

      // ✅ Save to Firestore under logged user
      try {
        await MarketCacheService.saveAvgByMarket(
          crop: _crop,
          days: _days,
          payload: res,
        );
      } catch (_) {
        // ignore save errors (user might not be logged)
      }

      if (!mounted) return;
      setState(() {
        _rows = rows.cast<Map<String, dynamic>>();
        _top = top.cast<Map<String, dynamic>>();
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  // ✅ Pull-to-refresh
  Future<void> _refresh() async {
    await _fetch();
  }

  // ✅ quick KPI values
  num _num(dynamic v) => (v is num) ? v : num.tryParse(v.toString()) ?? 0;

  @override
  Widget build(BuildContext context) {
    final bars = _rows
        .take(12)
        .map((r) => _BarPoint(
              (r["market"] ?? "").toString(),
              _num(r["avg_price"]).toDouble(),
            ))
        .toList();

    final top1 = _top.isNotEmpty ? _top.first : null;
    final bestMarket = top1 == null ? "—" : (top1["market"] ?? "—").toString();
    final bestAvg = top1 == null
        ? 0
        : _num(top1["avg_price"]).toDouble().toStringAsFixed(0);

    final totalMarkets = _rows.length;

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
                  right: -14,
                  top: 18,
                  child: Opacity(
                    opacity: 0.10,
                    child: Icon(Icons.auto_graph_rounded,
                        size: 160, color: Colors.white),
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
                          "Market",
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
                      "Compare & History",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.98),
                        fontWeight: FontWeight.w900,
                        fontSize: 22,
                      ),
                    ),
                    const SizedBox(height: 6),
                    Text(
                      "See best markets, averages and trends",
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
                        _HeaderChip(
                          icon: Icons.eco_rounded,
                          text: _crop,
                        ),
                        _HeaderChip(
                          icon: Icons.date_range_rounded,
                          text: "Last $_days days",
                        ),
                        _HeaderChip(
                          icon: Icons.storefront_rounded,
                          text: "$totalMarkets markets",
                        ),
                      ],
                    ),
                  ],
                ),
              ],
            ),
          ),

          const SizedBox(height: 14),

          // ===== BODY (Scrollable + pull to refresh) =====
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
                  padding: const EdgeInsets.fromLTRB(16, 18, 16, 18),
                  children: [
                    // ✅ Title + tiny hint
                    Row(
                      children: [
                        const Expanded(
                          child: Text(
                            "Market Comparison",
                            style: TextStyle(
                              fontWeight: FontWeight.w900,
                              fontSize: 18,
                              color: AppColors.textDark,
                            ),
                          ),
                        ),
                        Text(
                          "Pull to refresh",
                          style: TextStyle(
                            fontWeight: FontWeight.w800,
                            color: AppColors.textDark.withOpacity(0.55),
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 12),

                    // ✅ Controls row (Crop + Days) improved
                    Row(
                      children: [
                        Expanded(
                          child: _CropDropdown(
                            value: _crop,
                            onChanged: (v) async {
                              setState(() => _crop = v);
                              await _loadCacheThenFetch();
                            },
                          ),
                        ),
                        const SizedBox(width: 10),
                        _DaysChip(
                          days: _days,
                          onChanged: (d) async {
                            setState(() => _days = d);
                            await _loadCacheThenFetch();
                          },
                        )
                      ],
                    ),

                    const SizedBox(height: 12),

                    // ✅ Error banner
                    if (_error != null && !_loading)
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.red.withOpacity(0.06),
                          borderRadius: BorderRadius.circular(16),
                          border:
                              Border.all(color: Colors.red.withOpacity(0.20)),
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

                    // ✅ Loading line
                    if (_loading)
                      Padding(
                        padding: const EdgeInsets.only(top: 6, bottom: 6),
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
                              "Loading market data...",
                              style: TextStyle(
                                color: AppColors.textDark.withOpacity(0.70),
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ],
                        ),
                      ),

                    const SizedBox(height: 10),

                    // ✅ KPI STRIP (Best market + Avg price)
                    Container(
                      padding: const EdgeInsets.all(14),
                      decoration: BoxDecoration(
                        color: AppColors.primary.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: AppColors.border),
                      ),
                      child: Row(
                        children: [
                          Container(
                            width: 44,
                            height: 44,
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(16),
                              border: Border.all(color: AppColors.border),
                            ),
                            child: const Icon(Icons.emoji_events_rounded,
                                color: AppColors.darkGreen),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(
                                  "Best Market (Avg)",
                                  style: TextStyle(
                                    fontWeight: FontWeight.w900,
                                    color: AppColors.textDark,
                                  ),
                                ),
                                const SizedBox(height: 2),
                                Text(
                                  "$bestMarket • $bestAvg LKR/kg",
                                  style: TextStyle(
                                    fontWeight: FontWeight.w800,
                                    color: AppColors.textDark.withOpacity(0.70),
                                  ),
                                ),
                              ],
                            ),
                          ),
                          const Icon(Icons.arrow_forward_ios_rounded,
                              size: 16, color: AppColors.darkGreen),
                        ],
                      ),
                    ),

                    const SizedBox(height: 12),

                    // ===== TOP 3 cards (enhanced) =====
                    if (_top.isNotEmpty && !_loading)
                      Row(
                        children:
                            _top.take(3).toList().asMap().entries.map((e) {
                          final idx = e.key;
                          final t = e.value;

                          final m = (t["market"] ?? "").toString();
                          final p = _num(t["avg_price"]).toDouble();
                          final rank = idx + 1;

                          final badge = (rank == 1)
                              ? "Top"
                              : (rank == 2)
                                  ? "2nd"
                                  : "3rd";

                          return Expanded(
                            child: Container(
                              margin: EdgeInsets.only(right: idx == 2 ? 0 : 8),
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(18),
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
                                        padding: const EdgeInsets.symmetric(
                                            horizontal: 10, vertical: 6),
                                        decoration: BoxDecoration(
                                          color: AppColors.primary
                                              .withOpacity(0.18),
                                          borderRadius:
                                              BorderRadius.circular(999),
                                          border: Border.all(
                                            color: AppColors.primary
                                                .withOpacity(0.30),
                                          ),
                                        ),
                                        child: Text(
                                          badge,
                                          style: const TextStyle(
                                            fontWeight: FontWeight.w900,
                                            fontSize: 11.5,
                                            color: AppColors.darkGreen,
                                          ),
                                        ),
                                      ),
                                      const Spacer(),
                                      const Icon(Icons.star_rounded,
                                          size: 18, color: AppColors.darkGreen),
                                    ],
                                  ),
                                  const SizedBox(height: 8),
                                  Text(
                                    m,
                                    maxLines: 1,
                                    overflow: TextOverflow.ellipsis,
                                    style: const TextStyle(
                                      fontWeight: FontWeight.w900,
                                      fontSize: 12.8,
                                      color: AppColors.textDark,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    "${p.toStringAsFixed(0)} LKR/kg",
                                    style: TextStyle(
                                      fontWeight: FontWeight.w900,
                                      color:
                                          AppColors.textDark.withOpacity(0.72),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          );
                        }).toList(),
                      ),

                    const SizedBox(height: 12),

                    // ===== CHART CARD (more modern header) =====
                    Container(
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
                                child: const Icon(Icons.bar_chart_rounded,
                                    color: AppColors.darkGreen),
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(
                                  "$_crop • Avg Price by Market",
                                  style: const TextStyle(
                                    color: AppColors.textDark,
                                    fontWeight: FontWeight.w900,
                                    fontSize: 14.4,
                                  ),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 6),
                          Text(
                            "Last $_days days (top 12 markets)",
                            style: TextStyle(
                              color: AppColors.textDark.withOpacity(0.60),
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(height: 12),
                          _MiniBarChart(bars: bars),
                        ],
                      ),
                    ),

                    const SizedBox(height: 12),

                    // ===== TABLE (modern list) =====
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: AppColors.surface,
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: AppColors.border),
                      ),
                      child: _rows.isEmpty && !_loading
                          ? Center(
                              child: Padding(
                                padding: const EdgeInsets.all(10),
                                child: Text(
                                  "No data",
                                  style: TextStyle(
                                    color: AppColors.textDark.withOpacity(0.6),
                                    fontWeight: FontWeight.w800,
                                  ),
                                ),
                              ),
                            )
                          : Column(
                              children: [
                                Row(
                                  children: [
                                    Expanded(
                                      flex: 3,
                                      child: Text(
                                        "Market",
                                        style: TextStyle(
                                          fontWeight: FontWeight.w900,
                                          color: AppColors.textDark
                                              .withOpacity(0.65),
                                        ),
                                      ),
                                    ),
                                    Expanded(
                                      flex: 2,
                                      child: Text(
                                        "Avg",
                                        textAlign: TextAlign.right,
                                        style: TextStyle(
                                          fontWeight: FontWeight.w900,
                                          color: AppColors.textDark
                                              .withOpacity(0.65),
                                        ),
                                      ),
                                    ),
                                    Expanded(
                                      flex: 2,
                                      child: Text(
                                        "Min",
                                        textAlign: TextAlign.right,
                                        style: TextStyle(
                                          fontWeight: FontWeight.w900,
                                          color: AppColors.textDark
                                              .withOpacity(0.65),
                                        ),
                                      ),
                                    ),
                                    Expanded(
                                      flex: 2,
                                      child: Text(
                                        "Max",
                                        textAlign: TextAlign.right,
                                        style: TextStyle(
                                          fontWeight: FontWeight.w900,
                                          color: AppColors.textDark
                                              .withOpacity(0.65),
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                                Divider(
                                  height: 16,
                                  color: AppColors.border.withOpacity(0.9),
                                ),
                                ..._rows.take(10).map((r) {
                                  final market = (r["market"] ?? "").toString();
                                  final avg =
                                      _num(r["avg_price"]).toStringAsFixed(0);
                                  final min =
                                      _num(r["min_price"]).toStringAsFixed(0);
                                  final max =
                                      _num(r["max_price"]).toStringAsFixed(0);

                                  return Padding(
                                    padding: const EdgeInsets.only(bottom: 10),
                                    child: Container(
                                      padding: const EdgeInsets.all(12),
                                      decoration: BoxDecoration(
                                        color: Colors.white,
                                        borderRadius: BorderRadius.circular(16),
                                        border:
                                            Border.all(color: AppColors.border),
                                      ),
                                      child: Row(
                                        children: [
                                          Expanded(
                                            flex: 3,
                                            child: Text(
                                              market,
                                              maxLines: 1,
                                              overflow: TextOverflow.ellipsis,
                                              style: const TextStyle(
                                                fontWeight: FontWeight.w900,
                                                color: AppColors.textDark,
                                              ),
                                            ),
                                          ),
                                          Expanded(
                                            flex: 2,
                                            child: Text(
                                              avg,
                                              textAlign: TextAlign.right,
                                              style: const TextStyle(
                                                fontWeight: FontWeight.w900,
                                                color: AppColors.darkGreen,
                                              ),
                                            ),
                                          ),
                                          Expanded(
                                            flex: 2,
                                            child: Text(
                                              min,
                                              textAlign: TextAlign.right,
                                              style: TextStyle(
                                                fontWeight: FontWeight.w800,
                                                color: AppColors.textDark
                                                    .withOpacity(0.7),
                                              ),
                                            ),
                                          ),
                                          Expanded(
                                            flex: 2,
                                            child: Text(
                                              max,
                                              textAlign: TextAlign.right,
                                              style: TextStyle(
                                                fontWeight: FontWeight.w800,
                                                color: AppColors.textDark
                                                    .withOpacity(0.7),
                                              ),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  );
                                }).toList(),
                                if (_rows.length > 10) ...[
                                  const SizedBox(height: 2),
                                  Text(
                                    "Showing 10 of ${_rows.length}. Pull to refresh for latest.",
                                    style: TextStyle(
                                      fontWeight: FontWeight.w700,
                                      color:
                                          AppColors.textDark.withOpacity(0.55),
                                      fontSize: 12,
                                    ),
                                  ),
                                ],
                              ],
                            ),
                    ),

                    const SizedBox(height: 12),

                    // ✅ Historical Trends Button (kept)
                    SizedBox(
                      width: double.infinity,
                      height: 54,
                      child: ElevatedButton.icon(
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (_) => MarketHistoricalTrendsScreen(
                                initialCrop: _crop,
                              ),
                            ),
                          );
                        },
                        icon: const Icon(Icons.timeline_rounded),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppColors.primary,
                          foregroundColor: AppColors.darkGreen,
                          elevation: 0,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(18),
                          ),
                        ),
                        label: const Text(
                          "Historical Trends",
                          style: TextStyle(
                            fontWeight: FontWeight.w900,
                            fontSize: 15.5,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// ===== HEADER CHIP =====
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

// ===== Crop dropdown =====
class _CropDropdown extends StatelessWidget {
  final String value;
  final ValueChanged<String> onChanged;
  const _CropDropdown({required this.value, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    const crops = ["Rice", "Radish", "Red Onion", "Beetroot"];

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<String>(
          value: value,
          isExpanded: true,
          items: crops
              .map((c) => DropdownMenuItem(
                    value: c,
                    child: Text(
                      c,
                      style: const TextStyle(fontWeight: FontWeight.w800),
                    ),
                  ))
              .toList(),
          onChanged: (v) => onChanged(v ?? value),
        ),
      ),
    );
  }
}

// ===== Days selector =====
class _DaysChip extends StatelessWidget {
  final int days;
  final ValueChanged<int> onChanged;
  const _DaysChip({required this.days, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    final options = [7, 14, 30];

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: options.map((d) {
          final sel = d == days;
          return Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4),
            child: InkWell(
              borderRadius: BorderRadius.circular(12),
              onTap: () => onChanged(d),
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: sel
                      ? AppColors.primary.withOpacity(0.35)
                      : Colors.transparent,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: sel
                        ? AppColors.primary.withOpacity(0.6)
                        : AppColors.border,
                  ),
                ),
                child: Text(
                  "${d}D",
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    color: sel
                        ? AppColors.darkGreen
                        : AppColors.textDark.withOpacity(0.7),
                    fontSize: 12,
                  ),
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }
}

// ===== MINI BAR CHART =====
class _MiniBarChart extends StatelessWidget {
  final List<_BarPoint> bars;
  const _MiniBarChart({required this.bars});

  @override
  Widget build(BuildContext context) {
    if (bars.isEmpty) {
      return Container(
        height: 180,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.border),
        ),
        child: Text(
          "No data",
          style: TextStyle(
            color: AppColors.textDark.withOpacity(0.55),
            fontWeight: FontWeight.w700,
          ),
        ),
      );
    }

    final maxVal = bars.map((e) => e.value).reduce((a, b) => a > b ? a : b);

    return Container(
      height: 190,
      padding: const EdgeInsets.fromLTRB(10, 12, 10, 10),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Column(
        children: [
          Expanded(
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: bars.map((b) {
                final h =
                    maxVal <= 0 ? 0.05 : (b.value / maxVal).clamp(0.05, 1.0);

                return Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 6),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        Text(
                          b.value.toStringAsFixed(0),
                          style: TextStyle(
                            fontSize: 10.5,
                            fontWeight: FontWeight.w800,
                            color: AppColors.textDark.withOpacity(0.65),
                          ),
                        ),
                        const SizedBox(height: 6),
                        Expanded(
                          child: Align(
                            alignment: Alignment.bottomCenter,
                            child: FractionallySizedBox(
                              heightFactor: h,
                              widthFactor: 1,
                              child: Container(
                                decoration: BoxDecoration(
                                  color: AppColors.primary.withOpacity(0.35),
                                  borderRadius: BorderRadius.circular(10),
                                  border: Border.all(
                                    color: AppColors.primary.withOpacity(0.55),
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              }).toList(),
            ),
          ),
          const SizedBox(height: 10),
          Row(
            children: bars.map((b) {
              return Expanded(
                child: Center(
                  child: Text(
                    b.label,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: TextStyle(
                      fontSize: 10.5,
                      fontWeight: FontWeight.w800,
                      color: AppColors.textDark.withOpacity(0.60),
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }
}

class _BarPoint {
  final String label;
  final double value;
  const _BarPoint(this.label, this.value);
}
