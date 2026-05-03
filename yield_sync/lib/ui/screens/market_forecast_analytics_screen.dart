import 'dart:math';
import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../../services/market_api.dart';
import '../../services/forecast_db.dart';

class MarketForecastAnalyticsScreen extends StatefulWidget {
  final String crop;
  final String market;
  final int quantityKg;
  final int daysSinceHarvest;
  final int daysAhead;

  const MarketForecastAnalyticsScreen({
    super.key,
    required this.crop,
    required this.market,
    required this.quantityKg,
    required this.daysSinceHarvest,
    required this.daysAhead,
  });

  @override
  State<MarketForecastAnalyticsScreen> createState() =>
      _MarketForecastAnalyticsScreenState();
}

class _MarketForecastAnalyticsScreenState
    extends State<MarketForecastAnalyticsScreen> {
  bool _loading = true;
  String _err = "";
  bool _saved = false;

  Map<String, dynamic>? _history;

  final List<int> _horizons = const [7, 14, 30];
  List<Map<String, dynamic>> _rows = [];
  Map<String, dynamic>? _selected;

  @override
  void initState() {
    super.initState();
    _loadAll();
  }

  // ---------- format helpers ----------
  String _fmt2(num v) => v.toDouble().toStringAsFixed(2);
  String _fmtLkr(num v) => "LKR ${_fmt2(v)}";
  String _fmtPct(num v) => "${_fmt2(v)}%";
  String _fmtCompactLkr(num v) => "${_fmt2(v)} LKR";

  double _asDouble(dynamic v) =>
      (v is num) ? v.toDouble() : double.tryParse("$v") ?? 0.0;

  int _asInt(dynamic v) => (v is int) ? v : int.tryParse("$v") ?? 0;

  Future<void> _loadAll() async {
    setState(() {
      _loading = true;
      _err = "";
      _saved = false;
      _rows = [];
      _selected = null;
    });

    try {
      final history = await MarketApi.history(
        crop: widget.crop,
        market: widget.market,
        days: 30,
      );

      final List<Map<String, dynamic>> rows = [];
      double currentPrice = 0.0;

      for (final h in _horizons) {
        final predict = await MarketApi.predict(
          crop: widget.crop,
          market: widget.market,
          daysAhead: h,
        );

        final rec = await MarketApi.recommendation(
          crop: widget.crop,
          market: widget.market,
          daysAhead: h,
          quantityKg: widget.quantityKg.toDouble(),
        );

        currentPrice = currentPrice == 0.0
            ? _asDouble(predict["current_price"])
            : currentPrice;

        final predictedPrice = _asDouble(predict["predicted_price"]);
        final changeKg = predictedPrice - currentPrice;
        final changePct =
            currentPrice == 0 ? 0.0 : (changeKg / currentPrice) * 100.0;
        final total = changeKg * widget.quantityKg;

        final perKgFromRec = rec.containsKey("price_change_per_kg")
            ? _asDouble(rec["price_change_per_kg"])
            : changeKg;

        final totalFromRec = rec.containsKey("total_if_hold")
            ? _asDouble(rec["total_if_hold"])
            : total;

        rows.add({
          "horizon": h,
          "current_price": currentPrice,
          "predicted_price": predictedPrice,
          "change_per_kg": perKgFromRec,
          "change_pct": rec.containsKey("price_change_percent")
              ? _asDouble(rec["price_change_percent"])
              : changePct,
          "total": totalFromRec,
          "decision": (rec["decision"] ?? "NEUTRAL").toString(),
          "confidence": _asDouble(rec["confidence"] ?? 0),
          "shelf_life_days": _asInt(rec["shelf_life_days"] ?? 0),
          "reasoning": (rec["reasoning"] ?? "").toString(),
        });
      }

      // default selected from incoming daysAhead
      final selected = rows.firstWhere(
        (r) => r["horizon"] == widget.daysAhead,
        orElse: () => rows.first,
      );

      await ForecastDB.saveForecast(
        input: {
          "crop": widget.crop,
          "market": widget.market,
          "quantityKg": widget.quantityKg,
          "daysSinceHarvest": widget.daysSinceHarvest,
          "daysAhead": widget.daysAhead,
        },
        predict: {
          "current_price": selected["current_price"],
          "predicted_price": selected["predicted_price"],
          "price_change_percent": selected["change_pct"],
        },
        recommendation: {
          "decision": selected["decision"],
          "confidence": selected["confidence"],
          "shelf_life_days": selected["shelf_life_days"],
          "price_change_per_kg": selected["change_per_kg"],
          "total_if_hold": selected["total"],
          "reasoning": selected["reasoning"],
          "rows": rows,
        },
        history: history,
      );

      if (!mounted) return;
      setState(() {
        _history = history;
        _rows = rows;
        _selected = selected;
        _loading = false;
        _saved = true;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _err = "$e";
      });
    }
  }

  void _selectHorizon(int h) {
    if (_rows.isEmpty) return;
    final picked = _rows.firstWhere(
      (r) => _asInt(r["horizon"]) == h,
      orElse: () => _rows.first,
    );
    setState(() => _selected = picked);
  }

  List<double> _historyPrices() {
    final prices = (_history?["prices"] as List? ?? []);
    return prices.map((e) => _asDouble(e)).toList();
  }

  Color _decisionColor(String d) {
    final up = d.toUpperCase();
    if (up.contains("SELL")) return const Color(0xFFE05C5C);
    if (up.contains("HOLD")) return const Color(0xFF2FA36B);
    return const Color(0xFFC9A80B);
  }

  Color _dotColor(String d) {
    final up = d.toUpperCase();
    if (up.contains("SELL")) return const Color(0xFFE53935);
    if (up.contains("HOLD")) return const Color(0xFF2E7D32);
    return const Color(0xFFFFC107);
  }

  String _fallbackReason(double changePct) {
    if (changePct < 0) {
      return "Sell now, holding would lose ${_fmt2(changePct.abs())}%";
    }
    if (changePct > 0) {
      return "Holding may gain ${_fmt2(changePct.abs())}%";
    }
    return "Market looks stable, consider your storage & transport cost.";
  }

  @override
  Widget build(BuildContext context) {
    if (_loading) {
      return const Scaffold(
        backgroundColor: AppColors.surface,
        body: SafeArea(child: Center(child: CircularProgressIndicator())),
      );
    }

    if (_err.isNotEmpty) {
      return Scaffold(
        backgroundColor: AppColors.surface,
        body: SafeArea(
          child: Center(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Text(
                _err,
                style: const TextStyle(
                  color: Colors.red,
                  fontWeight: FontWeight.w800,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ),
      );
    }

    final sel = _selected ?? {};
    final decision = (sel["decision"] ?? "NEUTRAL").toString();
    final reasoning = (sel["reasoning"] ?? "").toString();

    final currentPrice = _asDouble(sel["current_price"]);
    final predictedPrice = _asDouble(sel["predicted_price"]);
    final changePct = _asDouble(sel["change_pct"]);
    final changeKg = _asDouble(sel["change_per_kg"]);
    final total = _asDouble(sel["total"]);
    final confidence = _asDouble(sel["confidence"]);
    final shelfLife = _asInt(sel["shelf_life_days"]);
    final selectedH = _asInt(sel["horizon"]);

    final List<int> xs = [0, ..._horizons];
    final List<double> ys = [
      currentPrice,
      ..._horizons.map((h) {
        final r = _rows.firstWhere((e) => _asInt(e["horizon"]) == h);
        return _asDouble(r["predicted_price"]);
      }).toList()
    ];

    final prices = _historyPrices();
    final decisionColor = _decisionColor(decision);

    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SafeArea(
        child: Column(
          children: [
            // ✅ Modern header + subtitle
            Container(
              width: double.infinity,
              padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
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
                    right: -12,
                    top: 26,
                    child: Opacity(
                      opacity: 0.10,
                      child: Icon(Icons.analytics_rounded,
                          size: 150, color: Colors.white),
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
                          Expanded(
                            child: Text(
                              "Forecast Analytics",
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.92),
                                fontWeight: FontWeight.w900,
                                fontSize: 16,
                              ),
                            ),
                          ),
                          CircleAvatar(
                            radius: 18,
                            backgroundColor: Colors.white.withOpacity(0.12),
                            child: const Icon(Icons.person_rounded,
                                color: Colors.white),
                          ),
                        ],
                      ),
                      const SizedBox(height: 6),
                      Text(
                        "${widget.crop} • ${widget.market}",
                        style: TextStyle(
                          color: Colors.white.withOpacity(0.80),
                          fontWeight: FontWeight.w700,
                          fontSize: 12.8,
                        ),
                      ),
                      const SizedBox(height: 10),

                      // ✅ Horizon chips (interactive)
                      _HorizonChips(
                        horizons: _horizons,
                        selected: selectedH == 0 ? widget.daysAhead : selectedH,
                        onTap: _selectHorizon,
                      ),

                      const SizedBox(height: 10),

                      // ✅ Saved pill / Refresh
                      Row(
                        children: [
                          if (_saved)
                            Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 10, vertical: 6),
                              decoration: BoxDecoration(
                                color: Colors.white.withOpacity(0.16),
                                borderRadius: BorderRadius.circular(999),
                                border: Border.all(
                                  color: Colors.white.withOpacity(0.25),
                                ),
                              ),
                              child: const Text(
                                "Saved to history ✅",
                                style: TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.w800,
                                  fontSize: 12,
                                ),
                              ),
                            ),
                          const Spacer(),
                          InkWell(
                            borderRadius: BorderRadius.circular(999),
                            onTap: _loadAll,
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 12, vertical: 8),
                              decoration: BoxDecoration(
                                color: Colors.white.withOpacity(0.12),
                                borderRadius: BorderRadius.circular(999),
                                border: Border.all(
                                  color: Colors.white.withOpacity(0.18),
                                ),
                              ),
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  const Icon(Icons.refresh_rounded,
                                      size: 18, color: Colors.white),
                                  const SizedBox(width: 8),
                                  Text(
                                    "Refresh",
                                    style: TextStyle(
                                      color: Colors.white.withOpacity(0.92),
                                      fontWeight: FontWeight.w900,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
            ),

            // ✅ Body
            Expanded(
              child: RefreshIndicator(
                onRefresh: _loadAll,
                child: ListView(
                  physics: const BouncingScrollPhysics(),
                  padding: const EdgeInsets.fromLTRB(16, 12, 16, 18),
                  children: [
                    // ✅ Decision hero card (interactive + readable)
                    Container(
                      padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
                      decoration: BoxDecoration(
                        color: decisionColor,
                        borderRadius: BorderRadius.circular(20),
                        boxShadow: [
                          BoxShadow(
                            color: decisionColor.withOpacity(0.22),
                            blurRadius: 22,
                            offset: const Offset(0, 14),
                          ),
                        ],
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Expanded(
                                child: Text(
                                  decision,
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontWeight: FontWeight.w900,
                                    fontSize: 28,
                                  ),
                                ),
                              ),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 10, vertical: 6),
                                decoration: BoxDecoration(
                                  color: Colors.white.withOpacity(0.16),
                                  borderRadius: BorderRadius.circular(999),
                                  border: Border.all(
                                    color: Colors.white.withOpacity(0.25),
                                  ),
                                ),
                                child: Text(
                                  "${selectedH}D",
                                  style: const TextStyle(
                                    color: Colors.white,
                                    fontWeight: FontWeight.w900,
                                    fontSize: 12.5,
                                  ),
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 8),
                          Text(
                            reasoning.trim().isNotEmpty
                                ? reasoning
                                : _fallbackReason(changePct),
                            style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w700,
                              height: 1.25,
                            ),
                          ),
                          const SizedBox(height: 12),
                          Row(
                            children: [
                              _miniPill("Now", "${_fmtLkr(currentPrice)}/kg"),
                              const SizedBox(width: 10),
                              _miniPill(
                                  "Pred", "${_fmtLkr(predictedPrice)}/kg"),
                              const SizedBox(width: 10),
                              _miniPill("Change", _fmtPct(changePct)),
                            ],
                          ),
                        ],
                      ),
                    ),

                    const SizedBox(height: 14),

                    // ✅ Top quick stats row
                    Row(
                      children: [
                        Expanded(
                          child: _QuickStat(
                            title: "Confidence",
                            value: _fmtPct(confidence),
                            icon: Icons.verified_rounded,
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: _QuickStat(
                            title: "Shelf Life",
                            value: "$shelfLife days",
                            icon: Icons.timelapse_rounded,
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 10),

                    Row(
                      children: [
                        Expanded(
                          child: _QuickStat(
                            title: "Change / kg",
                            value: _fmtCompactLkr(changeKg),
                            icon: Icons.swap_vert_rounded,
                            valueColor: changeKg >= 0
                                ? const Color(0xFF2FA36B)
                                : const Color(0xFFE05C5C),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: _QuickStat(
                            title: "Total (Hold)",
                            value: _fmtCompactLkr(total),
                            icon: Icons.payments_rounded,
                            valueColor: total >= 0
                                ? const Color(0xFF2FA36B)
                                : const Color(0xFFE05C5C),
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 12),

                    // ✅ Profit by horizon (tap to select)
                    _card(
                      title: "Profit Comparison",
                      subtitle: "Tap a card to change horizon",
                      child: Column(
                        children: _rows.map((r) {
                          final h = _asInt(r["horizon"]);
                          final pred = _asDouble(r["predicted_price"]);
                          final ckg = _asDouble(r["change_per_kg"]);
                          final pct = _asDouble(r["change_pct"]);
                          final tot = _asDouble(r["total"]);
                          final dec = (r["decision"] ?? "NEUTRAL").toString();
                          final isSelected = h == selectedH;

                          final bg = isSelected
                              ? AppColors.primary.withOpacity(0.14)
                              : Colors.white;

                          return InkWell(
                            borderRadius: BorderRadius.circular(16),
                            onTap: () => _selectHorizon(h),
                            child: Container(
                              margin: const EdgeInsets.only(bottom: 10),
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: bg,
                                borderRadius: BorderRadius.circular(16),
                                border: Border.all(
                                  color: isSelected
                                      ? AppColors.primary.withOpacity(0.55)
                                      : AppColors.border,
                                ),
                                boxShadow: [
                                  if (isSelected)
                                    BoxShadow(
                                      color:
                                          AppColors.primary.withOpacity(0.12),
                                      blurRadius: 18,
                                      offset: const Offset(0, 12),
                                    )
                                ],
                              ),
                              child: Column(
                                children: [
                                  Row(
                                    children: [
                                      Container(
                                        width: 34,
                                        height: 34,
                                        decoration: BoxDecoration(
                                          color: Colors.white,
                                          borderRadius:
                                              BorderRadius.circular(12),
                                          border: Border.all(
                                              color: AppColors.border),
                                        ),
                                        child: Icon(
                                          Icons.calendar_month_rounded,
                                          color: AppColors.darkGreen,
                                          size: 18,
                                        ),
                                      ),
                                      const SizedBox(width: 10),
                                      Text(
                                        "$h days",
                                        style: const TextStyle(
                                          fontWeight: FontWeight.w900,
                                          color: AppColors.textDark,
                                        ),
                                      ),
                                      const Spacer(),
                                      Container(
                                        width: 10,
                                        height: 10,
                                        decoration: BoxDecoration(
                                          color: _dotColor(dec),
                                          shape: BoxShape.circle,
                                        ),
                                      ),
                                      const SizedBox(width: 8),
                                      Text(
                                        dec,
                                        style: const TextStyle(
                                          fontWeight: FontWeight.w900,
                                          color: AppColors.textDark,
                                        ),
                                      ),
                                      if (isSelected) ...[
                                        const SizedBox(width: 10),
                                        Container(
                                          padding: const EdgeInsets.symmetric(
                                              horizontal: 10, vertical: 6),
                                          decoration: BoxDecoration(
                                            color: AppColors.darkGreen
                                                .withOpacity(0.10),
                                            borderRadius:
                                                BorderRadius.circular(999),
                                            border: Border.all(
                                              color: AppColors.darkGreen
                                                  .withOpacity(0.22),
                                            ),
                                          ),
                                          child: const Text(
                                            "Selected",
                                            style: TextStyle(
                                              color: AppColors.darkGreen,
                                              fontWeight: FontWeight.w900,
                                              fontSize: 12,
                                            ),
                                          ),
                                        ),
                                      ],
                                    ],
                                  ),
                                  const SizedBox(height: 10),
                                  _miniRow("Predicted", "${_fmtLkr(pred)} /kg"),
                                  const SizedBox(height: 6),
                                  _miniRow("Change/kg", _fmtCompactLkr(ckg)),
                                  const SizedBox(height: 6),
                                  _miniRow("Change %", _fmtPct(pct)),
                                  const SizedBox(height: 6),
                                  _miniRow(
                                    "Total (${widget.quantityKg}kg)",
                                    _fmtCompactLkr(tot),
                                  ),
                                ],
                              ),
                            ),
                          );
                        }).toList(),
                      ),
                    ),

                    const SizedBox(height: 12),

                    // ✅ Forecast chart
                    _card(
                      title: "Price Forecast",
                      subtitle: "Now vs future horizons",
                      child: SizedBox(
                        height: 210,
                        child: _ForecastChart(xDays: xs, prices: ys),
                      ),
                    ),

                    const SizedBox(height: 12),

                    // ✅ History chart
                    _card(
                      title: "Price History",
                      subtitle: "Last 30 days",
                      child: SizedBox(
                        height: 170,
                        child: _LineChart(values: prices),
                      ),
                    ),

                    const SizedBox(height: 12),

                    // ✅ Inputs
                    _card(
                      title: "Inputs Summary",
                      subtitle: "Used in recommendation",
                      child: Column(
                        children: [
                          _kv("Crop", widget.crop),
                          const SizedBox(height: 10),
                          _kv("Market", widget.market),
                          const SizedBox(height: 10),
                          _kv("Quantity", "${widget.quantityKg} kg"),
                          const SizedBox(height: 10),
                          _kv("Days Since Harvest",
                              "${widget.daysSinceHarvest} days"),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // -------------------- UI helpers --------------------
  Widget _miniPill(String k, String v) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.16),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.white.withOpacity(0.22)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              k,
              style: TextStyle(
                color: Colors.white.withOpacity(0.82),
                fontWeight: FontWeight.w800,
                fontSize: 11.5,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              v,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w900,
                fontSize: 13.5,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _card({
    required String title,
    required String subtitle,
    required Widget child,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 18,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: TextStyle(
              color: AppColors.textDark.withOpacity(0.75),
              fontWeight: FontWeight.w900,
              fontSize: 13.2,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            subtitle,
            style: TextStyle(
              color: AppColors.textDark.withOpacity(0.55),
              fontWeight: FontWeight.w700,
              fontSize: 12.2,
            ),
          ),
          const SizedBox(height: 12),
          child,
        ],
      ),
    );
  }

  Widget _kv(String k, String v) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              k,
              style: TextStyle(
                color: AppColors.textDark.withOpacity(0.70),
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
          Text(
            v,
            style: const TextStyle(
              color: AppColors.darkGreen,
              fontWeight: FontWeight.w900,
            ),
          ),
        ],
      ),
    );
  }

  Widget _miniRow(String k, String v) {
    return Row(
      children: [
        Expanded(
          child: Text(
            k,
            style: TextStyle(
              color: AppColors.textDark.withOpacity(0.70),
              fontWeight: FontWeight.w800,
            ),
          ),
        ),
        Text(
          v,
          style: const TextStyle(
            color: AppColors.darkGreen,
            fontWeight: FontWeight.w900,
          ),
        ),
      ],
    );
  }
}

// ✅ Horizon chips widget
class _HorizonChips extends StatelessWidget {
  final List<int> horizons;
  final int selected;
  final ValueChanged<int> onTap;

  const _HorizonChips({
    required this.horizons,
    required this.selected,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.10),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white.withOpacity(0.14)),
      ),
      child: Row(
        children: horizons.map((h) {
          final sel = h == selected;
          return Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4),
              child: InkWell(
                borderRadius: BorderRadius.circular(14),
                onTap: () => onTap(h),
                child: Container(
                  padding: const EdgeInsets.symmetric(vertical: 10),
                  decoration: BoxDecoration(
                    color: sel
                        ? Colors.white.withOpacity(0.20)
                        : Colors.transparent,
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(
                      color: sel
                          ? Colors.white.withOpacity(0.30)
                          : Colors.transparent,
                    ),
                  ),
                  child: Center(
                    child: Text(
                      "${h}D",
                      style: TextStyle(
                        color: Colors.white.withOpacity(sel ? 0.95 : 0.75),
                        fontWeight: FontWeight.w900,
                      ),
                    ),
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

class _QuickStat extends StatelessWidget {
  final String title;
  final String value;
  final IconData icon;
  final Color? valueColor;

  const _QuickStat({
    required this.title,
    required this.value,
    required this.icon,
    this.valueColor,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
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
      child: Row(
        children: [
          Container(
            width: 38,
            height: 38,
            decoration: BoxDecoration(
              color: AppColors.primary.withOpacity(0.16),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppColors.border),
            ),
            child: Icon(icon, color: AppColors.darkGreen, size: 20),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    fontWeight: FontWeight.w800,
                    color: AppColors.textDark.withOpacity(0.60),
                    fontSize: 11.8,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  value,
                  style: TextStyle(
                    fontWeight: FontWeight.w900,
                    color: valueColor ?? AppColors.textDark,
                    fontSize: 15.5,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ------------------- Forecast chart with axes -------------------
class _ForecastChart extends StatelessWidget {
  final List<int> xDays;
  final List<double> prices;
  const _ForecastChart({required this.xDays, required this.prices});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _ForecastPainter(xDays: xDays, prices: prices),
      child: const SizedBox.expand(),
    );
  }
}

class _ForecastPainter extends CustomPainter {
  final List<int> xDays;
  final List<double> prices;

  _ForecastPainter({required this.xDays, required this.prices});

  @override
  void paint(Canvas canvas, Size size) {
    if (prices.isEmpty || xDays.isEmpty || prices.length != xDays.length)
      return;

    const leftPad = 46.0;
    const rightPad = 14.0;
    const topPad = 14.0;
    const bottomPad = 34.0;

    final w = size.width - leftPad - rightPad;
    final h = size.height - topPad - bottomPad;

    final minV = prices.reduce(min);
    final maxV = prices.reduce(max);
    final range = (maxV - minV).abs() < 0.0001 ? 1.0 : (maxV - minV);

    final minX = xDays.reduce(min).toDouble();
    final maxX = xDays.reduce(max).toDouble();
    final xRange = (maxX - minX).abs() < 0.0001 ? 1.0 : (maxX - minX);

    Offset pt(int day, double price) {
      final x = leftPad + (w * ((day - minX) / xRange));
      final y = topPad + h - ((price - minV) / range) * h;
      return Offset(x, y);
    }

    final axisPaint = Paint()
      ..color = AppColors.textDark.withOpacity(0.18)
      ..strokeWidth = 1;

    final gridPaint = Paint()
      ..color = AppColors.textDark.withOpacity(0.08)
      ..strokeWidth = 1;

    final linePaint = Paint()
      ..color = AppColors.darkGreen
      ..strokeWidth = 2.6
      ..style = PaintingStyle.stroke;

    final dotPaint = Paint()..color = AppColors.primary;

    final origin = Offset(leftPad, topPad + h);
    final xEnd = Offset(leftPad + w, topPad + h);
    final yTop = Offset(leftPad, topPad);

    canvas.drawLine(origin, xEnd, axisPaint);
    canvas.drawLine(origin, yTop, axisPaint);

    const ySteps = 5;
    final tp = TextPainter(textDirection: TextDirection.ltr);

    for (int i = 0; i <= ySteps; i++) {
      final t = i / ySteps;
      final y = topPad + h - (h * t);

      canvas.drawLine(Offset(leftPad, y), Offset(leftPad + w, y), gridPaint);
      canvas.drawLine(Offset(leftPad - 4, y), Offset(leftPad, y), axisPaint);

      final value = minV + (range * t);
      tp.text = TextSpan(
        text: value.toStringAsFixed(0),
        style: TextStyle(
          color: Colors.black.withOpacity(0.65),
          fontSize: 10.5,
          fontWeight: FontWeight.w700,
        ),
      );
      tp.layout();
      tp.paint(canvas, Offset(leftPad - 8 - tp.width, y - tp.height / 2));
    }

    for (int i = 0; i < xDays.length; i++) {
      final day = xDays[i];
      final x = pt(day, minV).dx;

      canvas.drawLine(
          Offset(x, origin.dy), Offset(x, origin.dy + 4), axisPaint);

      tp.text = TextSpan(
        text: "$day",
        style: TextStyle(
          color: Colors.black.withOpacity(0.65),
          fontSize: 10.5,
          fontWeight: FontWeight.w700,
        ),
      );
      tp.layout();
      tp.paint(canvas, Offset(x - tp.width / 2, origin.dy + 6));
    }

    // Axis titles
    tp.text = TextSpan(
      text: "Price (LKR/kg)",
      style: TextStyle(
        color: Colors.black.withOpacity(0.75),
        fontSize: 11,
        fontWeight: FontWeight.w800,
      ),
    );
    tp.layout();
    canvas.save();
    canvas.translate(14, topPad + h / 2 + tp.width / 2);
    canvas.rotate(-pi / 2);
    tp.paint(canvas, const Offset(0, 0));
    canvas.restore();

    tp.text = TextSpan(
      text: "Days Ahead",
      style: TextStyle(
        color: Colors.black.withOpacity(0.75),
        fontSize: 11,
        fontWeight: FontWeight.w800,
      ),
    );
    tp.layout();
    tp.paint(canvas, Offset(leftPad + w / 2 - tp.width / 2, size.height - 18));

    // Line path
    final path = Path();
    for (int i = 0; i < prices.length; i++) {
      final p = pt(xDays[i], prices[i]);
      if (i == 0) {
        path.moveTo(p.dx, p.dy);
      } else {
        path.lineTo(p.dx, p.dy);
      }
    }
    canvas.drawPath(path, linePaint);

    // Points
    for (int i = 0; i < prices.length; i++) {
      final p = pt(xDays[i], prices[i]);
      canvas.drawCircle(p, 3.6, dotPaint);
    }

    // Red star at current (x=0)
    final starCenter = pt(0, prices.first);
    final starPaint = Paint()..color = const Color(0xFFE53935);

    Path starPath(Offset c, double rOuter, double rInner) {
      final p = Path();
      for (int i = 0; i < 10; i++) {
        final r = i.isEven ? rOuter : rInner;
        final a = -pi / 2 + (i * pi / 5);
        final x = c.dx + r * cos(a);
        final y = c.dy + r * sin(a);
        if (i == 0) {
          p.moveTo(x, y);
        } else {
          p.lineTo(x, y);
        }
      }
      p.close();
      return p;
    }

    canvas.drawPath(starPath(starCenter, 9, 4.5), starPaint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

// ------------------- History chart (unchanged) -------------------
class _LineChart extends StatelessWidget {
  final List<double> values;
  const _LineChart({required this.values});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _LineChartPainter(values),
      child: const SizedBox.expand(),
    );
  }
}

class _LineChartPainter extends CustomPainter {
  final List<double> v;
  _LineChartPainter(this.v);

  @override
  void paint(Canvas canvas, Size size) {
    if (v.isEmpty) return;

    final pad = 16.0;
    final w = size.width - pad * 2;
    final h = size.height - pad * 2;

    final minV = v.reduce(min);
    final maxV = v.reduce(max);
    final range = (maxV - minV).abs() < 0.0001 ? 1.0 : (maxV - minV);

    final axisPaint = Paint()
      ..color = AppColors.textDark.withOpacity(0.10)
      ..strokeWidth = 1;

    canvas.drawLine(
      Offset(pad, pad + h),
      Offset(pad + w, pad + h),
      axisPaint,
    );

    final linePaint = Paint()
      ..color = AppColors.darkGreen
      ..strokeWidth = 2.2
      ..style = PaintingStyle.stroke;

    final path = Path();
    for (int i = 0; i < v.length; i++) {
      final x = pad + (w * (i / max(1, (v.length - 1))));
      final y = pad + h - ((v[i] - minV) / range) * h;
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(path, linePaint);

    final dotPaint = Paint()..color = AppColors.primary;
    for (int i = 0; i < v.length; i++) {
      final x = pad + (w * (i / max(1, (v.length - 1))));
      final y = pad + h - ((v[i] - minV) / range) * h;
      canvas.drawCircle(Offset(x, y), 3.2, dotPaint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
