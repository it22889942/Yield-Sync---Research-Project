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
  final double? transportCostPerKg;
  final double? storageCostPerKgDay;
  final double? fixedCostTotal;

  const MarketForecastAnalyticsScreen({
    super.key,
    required this.crop,
    required this.market,
    required this.quantityKg,
    required this.daysSinceHarvest,
    required this.daysAhead,
    this.transportCostPerKg,
    this.storageCostPerKgDay,
    this.fixedCostTotal,
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
  Map<String, dynamic>? _tomorrow;

  @override
  void initState() {
    super.initState();
    _loadAll();
  }

  // ---------- format helpers ----------
  String _fmt1(num v) => v.toDouble().toStringAsFixed(1);
  String _fmt2(num v) => v.toDouble().toStringAsFixed(2);
  String _fmtLkr(num v) => "LKR ${_fmt2(v)}";
  String _fmtPct(num v) => "${_fmt2(v)}%";
  String _fmtCompactLkr(num v) => "${_fmt2(v)} LKR";

  double _asDouble(dynamic v) =>
      (v is num) ? v.toDouble() : double.tryParse("$v") ?? 0.0;

  int _asInt(dynamic v) => (v is int) ? v : int.tryParse("$v") ?? 0;

  int _defaultShelfLifeForCrop(String crop) {
    switch (crop.trim().toLowerCase()) {
      case "rice":
        return 180;
      case "beetroot":
        return 7;
      case "radish":
        return 5;
      case "red onion":
        return 30;
      default:
        return 30;
    }
  }

  int _resolveShelfLifeDays(Map<String, dynamic> rec) {
    final direct = _asInt(rec["shelf_life_days"] ?? rec["shelf_life"]);
    if (direct > 0) return direct;

    final fromPrediction = _asInt(
      (rec["prediction"] is Map)
          ? (rec["prediction"] as Map)["shelf_life_days"]
          : null,
    );
    if (fromPrediction > 0) return fromPrediction;

    return _defaultShelfLifeForCrop(widget.crop);
  }

  Future<void> _loadAll() async {
    setState(() {
      _loading = true;
      _err = "";
      _saved = false;
      _rows = [];
      _selected = null;
      _tomorrow = null;
    });

    try {
      final history = await MarketApi.history(
        crop: widget.crop,
        market: widget.market,
        days: 30,
      );

      final List<Map<String, dynamic>> rows = [];
      double currentPrice = 0.0;

      // Fetch tomorrow (1-day) data for the hero card — always fixed, not selectable
      Map<String, dynamic> tomorrowRow = {};
      {
        const th = 1;
        final tPredict = await MarketApi.predict(
          crop: widget.crop,
          market: widget.market,
          daysAhead: th,
        );
        final tRec = await MarketApi.recommendation(
          crop: widget.crop,
          market: widget.market,
          daysAhead: th,
          quantityKg: widget.quantityKg.toDouble(),
          daysSinceHarvest: widget.daysSinceHarvest,
          transportCostPerKg: widget.transportCostPerKg,
          storageCostPerKgDay: widget.storageCostPerKgDay,
          fixedCostTotal: widget.fixedCostTotal,
        );
        currentPrice = _asDouble(tPredict["current_price"]);
        final tPredictedPrice = _asDouble(tPredict["predicted_price"]);
        final tChangeKg = tPredictedPrice - currentPrice;
        final tChangePct = currentPrice == 0 ? 0.0 : (tChangeKg / currentPrice) * 100.0;
        final tTotal = tChangeKg * widget.quantityKg;
        final tPerKg = tRec.containsKey("price_change_per_kg") ? _asDouble(tRec["price_change_per_kg"]) : tChangeKg;
        final tTotalRec = tRec.containsKey("total_if_hold") ? _asDouble(tRec["total_if_hold"]) : tTotal;
        final tShelfLife = _resolveShelfLifeDays(tRec);
        final tRemainingShelfLife = tRec.containsKey("remaining_shelf_life_days")
            ? _asInt(tRec["remaining_shelf_life_days"])
            : (tShelfLife - widget.daysSinceHarvest).clamp(0, tShelfLife);
        final tProfitAnalysisMap = (tRec["profit_analysis"] as Map?)?.cast<String, dynamic>() ?? {};
        final tChangePctRec = tProfitAnalysisMap.containsKey("profit_change_percent")
            ? _asDouble(tProfitAnalysisMap["profit_change_percent"])
            : (tRec.containsKey("price_change_percent") ? _asDouble(tRec["price_change_percent"]) : tChangePct);
        final tEffectiveHoldDays = tRec.containsKey("effective_hold_days") ? _asInt(tRec["effective_hold_days"]) : 1;
        final tBackendDecision = (tRec["decision"] ?? "NEUTRAL").toString();
        final tDecision = _resolveDecision(
          backendDecision: tBackendDecision,
          changePct: tChangePctRec,
          totalIfHold: tTotalRec,
          remainingShelfLifeDays: tRemainingShelfLife,
          effectiveHoldDays: tEffectiveHoldDays,
        );
        final tBackendReasoning = (tRec["reasoning"] ?? "").toString().trim();
        final tReasoning = tDecision == tBackendDecision.toUpperCase() ? tBackendReasoning : _reasoningForDecision(tDecision, tChangePctRec);
        final tExpectedPrice = tRec.containsKey("expected_price") ? _asDouble(tRec["expected_price"]) : tPredictedPrice;
        final tBestTimingText = (tRec["best_timing_label"] ?? tRec["best_timing"] ?? "").toString();
        final tBestTimingDays = tRec.containsKey("best_timing_days")
            ? _asInt(tRec["best_timing_days"])
            : (tRemainingShelfLife <= 0 ? 0 : (tDecision.toUpperCase().contains("SELL") ? 0 : th));
        final tIsExpired = tRec["is_expired"] == true || tRemainingShelfLife <= 0;
        tomorrowRow = {
          "horizon": th,
          "current_price": currentPrice,
          "predicted_price": tPredictedPrice,
          "change_per_kg": tPerKg,
          "change_pct": tChangePctRec,
          "total": tTotalRec,
          "decision": tDecision,
          "confidence": _asDouble(tRec["confidence"] ?? 0),
          "shelf_life_days": tShelfLife,
          "warnings": (tRec["warnings"] as List? ?? []).map((e) => e.toString()).toList(),
          "reasoning": tReasoning,
          "expected_price": tExpectedPrice,
          "best_timing_text": tBestTimingText,
          "best_timing_days": tBestTimingDays,
          "is_expired": tIsExpired,
          "cost_breakdown": (tRec["cost_breakdown"] as Map?)?.cast<String, dynamic>() ?? {},
          "profit_analysis": (tRec["profit_analysis"] as Map?)?.cast<String, dynamic>() ?? {},
        };
      }

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
          daysSinceHarvest: widget.daysSinceHarvest,
          transportCostPerKg: widget.transportCostPerKg,
          storageCostPerKgDay: widget.storageCostPerKgDay,
          fixedCostTotal: widget.fixedCostTotal,
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
        final resolvedShelfLife = _resolveShelfLifeDays(rec);

        final costBreakdown =
            (rec["cost_breakdown"] as Map?)?.cast<String, dynamic>() ?? {};
        final profitAnalysis =
            (rec["profit_analysis"] as Map?)?.cast<String, dynamic>() ?? {};

        final resolvedChangePct = profitAnalysis.containsKey("profit_change_percent")
            ? _asDouble(profitAnalysis["profit_change_percent"])
            : (rec.containsKey("price_change_percent") ? _asDouble(rec["price_change_percent"]) : changePct);

        final remainingShelfLife = rec.containsKey("remaining_shelf_life_days")
            ? _asInt(rec["remaining_shelf_life_days"])
            : (resolvedShelfLife - widget.daysSinceHarvest).clamp(0, resolvedShelfLife);
        final effectiveHoldDays = rec.containsKey("effective_hold_days")
            ? _asInt(rec["effective_hold_days"])
            : h;

        final backendDecision = (rec["decision"] ?? "NEUTRAL").toString();
        final resolvedDecision = _resolveDecision(
          backendDecision: backendDecision,
          changePct: resolvedChangePct,
          totalIfHold: totalFromRec,
          remainingShelfLifeDays: remainingShelfLife,
          effectiveHoldDays: effectiveHoldDays,
        );
        final backendReasoning = (rec["reasoning"] ?? "").toString().trim();
        final resolvedReasoning = resolvedDecision == backendDecision.toUpperCase()
            ? backendReasoning
            : _reasoningForDecision(resolvedDecision, resolvedChangePct);

        final expectedPriceFromRec = rec.containsKey("expected_price")
            ? _asDouble(rec["expected_price"])
            : predictedPrice;

        final String bestTimingText = (rec["best_timing_label"] ?? rec["best_timing"] ?? "").toString();
        final int bestTimingDays = rec.containsKey("best_timing_days")
            ? _asInt(rec["best_timing_days"])
            : (remainingShelfLife <= 0
                ? 0
                : (resolvedDecision.toUpperCase().contains("SELL") ? 0 : h));
        final bool isExpired = rec["is_expired"] == true || remainingShelfLife <= 0;
        final warnings = (rec["warnings"] as List? ?? [])
            .map((e) => e.toString())
            .toList();

        rows.add({
          "horizon": h,
          "current_price": currentPrice,
          "predicted_price": predictedPrice,
          "change_per_kg": perKgFromRec,
          "change_pct": resolvedChangePct,
          "total": totalFromRec,
          "decision": resolvedDecision,
          "confidence": _asDouble(rec["confidence"] ?? 0),
          "shelf_life_days": resolvedShelfLife,
          "remaining_shelf_life_days": remainingShelfLife,
          "effective_hold_days": effectiveHoldDays,
          "warnings": warnings,
          "cost_breakdown": costBreakdown,
          "profit_analysis": profitAnalysis,
          "reasoning": resolvedReasoning,
          "expected_price": expectedPriceFromRec,
          "best_timing_text": bestTimingText,
          "best_timing_days": bestTimingDays,
          "is_expired": isExpired,
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
        _tomorrow = tomorrowRow;
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

  String _decisionFromChangePct(double changePct) {
    if (changePct >= 10.0) return "STRONG HOLD";
    if (changePct >= 2.0) return "HOLD";
    if (changePct <= -10.0) return "STRONG SELL";
    if (changePct <= -2.0) return "SELL";
    return "NEUTRAL";
  }

  String _resolveDecision({
    required String backendDecision,
    required double changePct,
    required double totalIfHold,
    int remainingShelfLifeDays = 999,
    int effectiveHoldDays = 999,
  }) {
    final backend = backendDecision.trim().toUpperCase();
    if (backend.isEmpty) return _decisionFromChangePct(changePct);

    // Shelf-life-based decisions are absolute — never override them
    if (remainingShelfLifeDays <= 0 || effectiveHoldDays == 0) return backend;

    final byChange = _decisionFromChangePct(changePct);
    final backendHold = backend.contains("HOLD");
    final backendSell = backend.contains("SELL");
    final metricsHold = changePct > 0 && totalIfHold > 0;
    final metricsSell = changePct < 0 && totalIfHold < 0;

    if ((backendSell && metricsHold) || (backendHold && metricsSell)) {
      return byChange;
    }
    return backend;
  }

  String _reasoningForDecision(String decision, double changePct) {
    final pct = _fmt1(changePct.abs());
    final signed = "${changePct >= 0 ? "+" : ""}${_fmt1(changePct)}";
    final up = decision.toUpperCase();
    if (up == "STRONG HOLD") return "Wait for +$pct% profit";
    if (up == "HOLD") return "Moderate profit opportunity: +$pct%";
    if (up == "STRONG SELL") return "Sell now to avoid $pct% loss";
    if (up == "SELL") return "Sell now, holding would lose $pct%";
    return "Price stable ($signed%), sell when convenient";
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

    // Tomorrow data — drives the hero decision card (fixed, never changes with horizon)
    final tmr = _tomorrow ?? {};
    final tmrDecision = (tmr["decision"] ?? "NEUTRAL").toString();
    final tmrReasoning = (tmr["reasoning"] ?? "").toString();
    final tmrCurrentPrice = _asDouble(tmr["current_price"]);
    final tmrPredictedPrice = _asDouble(tmr["predicted_price"]);
    final tmrChangePct = _asDouble(tmr["change_pct"]);
    final tmrShelfLife = _asInt(tmr["shelf_life_days"]);
    final tmrExpectedPrice = _asDouble(tmr["expected_price"]);
    final tmrBestTimingText = (tmr["best_timing_text"] ?? "").toString().trim();
    final tmrBestTimingDays = _asInt(tmr["best_timing_days"]);
    final tmrBestTimingLabel = tmrShelfLife <= 1
        ? "Sell Now"
        : (tmrBestTimingText.isNotEmpty
            ? tmrBestTimingText
            : (tmrBestTimingDays >= 0
                ? (tmrBestTimingDays == 0 ? "Sell Now" : "Hold for $tmrBestTimingDays days")
                : (tmrDecision.toUpperCase().contains("SELL") ? "Sell Now" : "Hold until Tomorrow")));
    final tmrWarnings = (tmr["warnings"] as List? ?? []).map((e) => e.toString()).toList();
    final tmrCostBreakdown = (tmr["cost_breakdown"] as Map?)?.cast<String, dynamic>() ?? {};
    final tmrProfitAnalysis = (tmr["profit_analysis"] as Map?)?.cast<String, dynamic>() ?? {};
    final tmrExpenditure = _asDouble(tmrCostBreakdown["transport_cost_now"]) +
        _asDouble(tmrCostBreakdown["storage_cost_total"]) +
        _asDouble(tmrCostBreakdown["fixed_cost_total"]);
    final tmrProfit = tmrProfitAnalysis.containsKey("revenue_if_sell_now")
        ? _asDouble(tmrProfitAnalysis["revenue_if_sell_now"])
        : 0.0;

    // Selected horizon data — drives quick stats, cost/profit cards, chart
    final sel = _selected ?? {};
    final changeKg = _asDouble(sel["change_per_kg"]);
    final total = _asDouble(sel["total"]);
    final confidence = _asDouble(sel["confidence"]);
    final shelfLife = _asInt(sel["shelf_life_days"]);
    final remainingShelfLife = _asInt(sel["remaining_shelf_life_days"] ?? shelfLife);
    final effectiveHoldDays = _asInt(sel["effective_hold_days"] ?? 0);
    final warnings = (sel["warnings"] as List? ?? []).map((e) => e.toString()).toList();
    final costBreakdown = (sel["cost_breakdown"] as Map?)?.cast<String, dynamic>() ?? {};
    final profitAnalysis = (sel["profit_analysis"] as Map?)?.cast<String, dynamic>() ?? {};
    final selectedH = _asInt(sel["horizon"]);
    final selectedStorageCost = costBreakdown.containsKey("storage_cost_per_kg_day")
        ? _asDouble(costBreakdown["storage_cost_per_kg_day"]) * selectedH * widget.quantityKg
        : _asDouble(costBreakdown["storage_cost_total"]);
    final expenditure = _asDouble(costBreakdown["transport_cost_now"]) +
        selectedStorageCost +
        _asDouble(costBreakdown["fixed_cost_total"]);
    final profitNow = _asDouble(profitAnalysis["revenue_if_hold"]);
    final currentPrice = _asDouble(sel["current_price"]);

    final List<int> xs = [0, ..._horizons];
    final List<double> ys = [
      currentPrice,
      ..._horizons.map((h) {
        final r = _rows.firstWhere((e) => _asInt(e["horizon"]) == h);
        return _asDouble(r["predicted_price"]);
      }).toList()
    ];

    final decisionColor = _decisionColor(tmrDecision);

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
                                  tmrDecision,
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
                                child: const Text(
                                  "Tomorrow",
                                  style: TextStyle(
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
                            tmrReasoning.trim().isNotEmpty
                                ? tmrReasoning
                                : _fallbackReason(tmrChangePct),
                            style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w700,
                              height: 1.25,
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            "Best timing: $tmrBestTimingLabel",
                            style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w800,
                              height: 1.25,
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            "Expected price: ${_fmt2(tmrExpectedPrice)} LKR/kg",
                            style: const TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.w800,
                              height: 1.25,
                            ),
                          ),
                          const SizedBox(height: 12),
                          Row(
                            children: [
                              _miniPill("Tomorrow", "${_fmtLkr(tmrCurrentPrice)}/kg"),
                              const SizedBox(width: 10),
                              _miniPill("Profit", _fmtLkr(tmrProfit)),
                              const SizedBox(width: 10),
                              _miniPill("Expenditure", _fmtLkr(tmrExpenditure)),
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
                            value: _fmtPct(confidence > 1 ? confidence : confidence * 100),
                            icon: Icons.verified_rounded,
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: _QuickStat(
                            title: "Shelf Life",
                            value: "$remainingShelfLife days",
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
                            value: remainingShelfLife <= 0 ? "N/A" : _fmtCompactLkr(changeKg),
                            icon: Icons.swap_vert_rounded,
                            valueColor: remainingShelfLife <= 0
                                ? AppColors.textDark
                                : (changeKg >= 0
                                    ? const Color(0xFF2FA36B)
                                    : const Color(0xFFE05C5C)),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: _QuickStat(
                            title: "Total (Hold)",
                            value: remainingShelfLife <= 0 ? "N/A" : _fmtCompactLkr(total),
                            icon: Icons.payments_rounded,
                            valueColor: remainingShelfLife <= 0
                                ? AppColors.textDark
                                : (total >= 0
                                    ? const Color(0xFF2FA36B)
                                    : const Color(0xFFE05C5C)),
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 10),

                    Row(
                      children: [
                        Expanded(
                          child: _QuickStat(
                            title: "Expenditure",
                            value: expenditure > 0
                                ? _fmtCompactLkr(expenditure)
                                : "N/A",
                            icon: Icons.account_balance_wallet_rounded,
                            valueColor: const Color(0xFFE05C5C),
                          ),
                        ),
                        const SizedBox(width: 10),
                        Expanded(
                          child: _QuickStat(
                            title: "Profit",
                            value: remainingShelfLife <= 0
                                ? "N/A"
                                : (profitAnalysis.isNotEmpty
                                    ? _fmtCompactLkr(profitNow)
                                    : "N/A"),
                            icon: Icons.trending_up_rounded,
                            valueColor: remainingShelfLife <= 0
                                ? AppColors.textDark
                                : (profitNow >= 0
                                    ? const Color(0xFF2FA36B)
                                    : const Color(0xFFE05C5C)),
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
                                      color: AppColors.primary.withOpacity(0.12),
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
                                          borderRadius: BorderRadius.circular(12),
                                          border: Border.all(color: AppColors.border),
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
                                      if (isSelected) ...[
                                        const SizedBox(width: 10),
                                        Container(
                                          padding: const EdgeInsets.symmetric(
                                              horizontal: 10, vertical: 6),
                                          decoration: BoxDecoration(
                                            color: AppColors.darkGreen.withOpacity(0.10),
                                            borderRadius: BorderRadius.circular(999),
                                            border: Border.all(
                                              color: AppColors.darkGreen.withOpacity(0.22),
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

                    // ✅ Tomorrow warnings (from the hero card data)
                    if (tmrWarnings.isNotEmpty) ...[
                      Container(
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: const Color(0xFFFFF3E0),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: const Color(0xFFFFB74D)),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                const Icon(Icons.warning_amber_rounded,
                                    color: Color(0xFFE65100), size: 20),
                                const SizedBox(width: 8),
                                const Text(
                                  "TOMORROW WARNINGS",
                                  style: TextStyle(
                                    fontWeight: FontWeight.w900,
                                    color: Color(0xFFE65100),
                                    fontSize: 13,
                                    letterSpacing: 1,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 10),
                            ...tmrWarnings.map((w) => Padding(
                                  padding: const EdgeInsets.only(bottom: 6),
                                  child: Row(
                                    crossAxisAlignment: CrossAxisAlignment.start,
                                    children: [
                                      const Text("• ",
                                          style: TextStyle(
                                              color: Color(0xFFE65100),
                                              fontWeight: FontWeight.w900)),
                                      Expanded(
                                        child: Text(
                                          w,
                                          style: const TextStyle(
                                            color: Color(0xFFBF360C),
                                            fontWeight: FontWeight.w700,
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                )),
                          ],
                        ),
                      ),
                      const SizedBox(height: 12),
                    ],

                    // ✅ Warnings card (only shown if backend returned warnings)
                    if (warnings.isNotEmpty) ...[
                      Container(
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: const Color(0xFFFFF3E0),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: const Color(0xFFFFB74D)),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                const Icon(Icons.warning_amber_rounded,
                                    color: Color(0xFFE65100), size: 20),
                                const SizedBox(width: 8),
                                const Text(
                                  "WARNINGS",
                                  style: TextStyle(
                                    fontWeight: FontWeight.w900,
                                    color: Color(0xFFE65100),
                                    fontSize: 13,
                                    letterSpacing: 1,
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 10),
                            ...warnings.map((w) => Padding(
                                  padding: const EdgeInsets.only(bottom: 6),
                                  child: Row(
                                    crossAxisAlignment:
                                        CrossAxisAlignment.start,
                                    children: [
                                      const Text("• ",
                                          style: TextStyle(
                                              color: Color(0xFFE65100),
                                              fontWeight: FontWeight.w900)),
                                      Expanded(
                                        child: Text(
                                          w,
                                          style: const TextStyle(
                                            color: Color(0xFFBF360C),
                                            fontWeight: FontWeight.w700,
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                )),
                          ],
                        ),
                      ),
                      const SizedBox(height: 12),
                    ],

                    // ✅ Perishability card
                    _card(
                      title: "Perishability",
                      subtitle: "Shelf life & safe hold period",
                      child: Column(
                        children: [
                          _kv("Shelf Life", "$shelfLife days"),
                          const SizedBox(height: 10),
                          _kv("Days Since Harvest",
                              "${widget.daysSinceHarvest} days"),
                          const SizedBox(height: 10),
                          _kv("Remaining Shelf Life",
                              "$remainingShelfLife days"),
                          const SizedBox(height: 10),
                          _kv("Effective Hold Days",
                              "$effectiveHoldDays days"),
                        ],
                      ),
                    ),

                    const SizedBox(height: 12),

                    // ✅ Cost breakdown card (only shown if data available)
                    if (costBreakdown.isNotEmpty) ...[
                      _card(
                        title: "Cost Breakdown",
                        subtitle: "Transport, storage & fixed costs",
                        child: Column(
                          children: [
                            if (costBreakdown["transport_cost_per_kg"] != null)
                              _kv("Transport / kg",
                                  "${_fmtLkr(_asDouble(costBreakdown["transport_cost_per_kg"]))}"),
                            if (costBreakdown["transport_cost_per_kg"] != null)
                              const SizedBox(height: 10),
                            if (costBreakdown["storage_cost_per_kg_day"] != null)
                              _kv("Storage / kg / day",
                                  "${_fmtLkr(_asDouble(costBreakdown["storage_cost_per_kg_day"]))}"),
                            if (costBreakdown["storage_cost_per_kg_day"] != null)
                              const SizedBox(height: 10),
                            if (costBreakdown["storage_cost_total"] != null)
                              _kv("Total Storage Cost",
                                  _fmtLkr(selectedStorageCost)),
                            if (costBreakdown["storage_cost_total"] != null)
                              const SizedBox(height: 10),
                            if (costBreakdown["transport_cost_now"] != null)
                              _kv("Transport (Sell Now)",
                                  _fmtLkr(_asDouble(costBreakdown["transport_cost_now"]))),
                            if (costBreakdown["transport_cost_now"] != null)
                              const SizedBox(height: 10),
                            if (costBreakdown["transport_cost_later"] != null)
                              _kv("Transport (If Hold)",
                                  _fmtLkr(_asDouble(costBreakdown["transport_cost_later"]))),
                            if (costBreakdown["fixed_cost_total"] != null &&
                                _asDouble(costBreakdown["fixed_cost_total"]) > 0) ...[
                              const SizedBox(height: 10),
                              _kv("Fixed Cost",
                                  _fmtLkr(_asDouble(costBreakdown["fixed_cost_total"]))),
                            ],
                          ],
                        ),
                      ),
                      const SizedBox(height: 12),
                    ],

                    // ✅ Profit analysis card (only shown if data available)
                    if (profitAnalysis.isNotEmpty) ...[
                      _card(
                        title: "Profit Analysis",
                        subtitle: "Revenue & profit comparison",
                        child: Column(
                          children: [
                            if (profitAnalysis["revenue_if_sell_now"] != null)
                              _kv("Revenue (Sell Now)",
                                  _fmtLkr(_asDouble(profitAnalysis["revenue_if_sell_now"]))),
                            if (profitAnalysis["revenue_if_sell_now"] != null)
                              const SizedBox(height: 10),
                            if (profitAnalysis["revenue_if_hold"] != null)
                              _kv("Revenue (If Hold)",
                                  _fmtLkr(_asDouble(profitAnalysis["revenue_if_hold"]))),
                            if (profitAnalysis["revenue_if_hold"] != null)
                              const SizedBox(height: 10),
                            if (profitAnalysis["profit_difference"] != null)
                              _kv("Profit Difference",
                                  _fmtLkr(_asDouble(profitAnalysis["profit_difference"]))),
                            if (profitAnalysis["profit_difference"] != null)
                              const SizedBox(height: 10),
                            if (profitAnalysis["profit_change_percent"] != null)
                              _kv("Profit Change",
                                  _fmtPct(_asDouble(profitAnalysis["profit_change_percent"]))),
                            if (profitAnalysis["spoilage_loss_percent"] != null) ...[
                              const SizedBox(height: 10),
                              _kv("Spoilage Loss",
                                  _fmtPct(_asDouble(profitAnalysis["spoilage_loss_percent"]))),
                            ],
                          ],
                        ),
                      ),
                      const SizedBox(height: 12),
                    ],

                    // ✅ Forecast chart
                    _card(
                      title: "Price Forecast",
                      subtitle: "Now vs future horizons",
                      child: SizedBox(
                        height: 210,
                        child: _ForecastChart(xDays: xs, prices: ys, selectedDay: selectedH),
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
  final int selectedDay;
  const _ForecastChart({required this.xDays, required this.prices, required this.selectedDay});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _ForecastPainter(xDays: xDays, prices: prices, selectedDay: selectedDay),
      child: const SizedBox.expand(),
    );
  }
}

class _ForecastPainter extends CustomPainter {
  final List<int> xDays;
  final List<double> prices;
  final int selectedDay;

  _ForecastPainter({required this.xDays, required this.prices, required this.selectedDay});

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

    // Highlight selected horizon
    final selIdx = xDays.indexOf(selectedDay);
    if (selIdx >= 0) {
      final selPt = pt(xDays[selIdx], prices[selIdx]);
      canvas.drawCircle(selPt, 9, Paint()..color = AppColors.primary.withOpacity(0.20));
      canvas.drawCircle(selPt, 5.5, Paint()..color = AppColors.primary);
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
