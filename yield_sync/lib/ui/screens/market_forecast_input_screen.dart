import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../widgets/app_shell.dart';
import '../../services/market_api.dart';
import 'market_forecast_analytics_screen.dart';

class MarketForecastInputScreen extends StatefulWidget {
  const MarketForecastInputScreen({super.key});

  @override
  State<MarketForecastInputScreen> createState() =>
      _MarketForecastInputScreenState();
}

class _MarketForecastInputScreenState extends State<MarketForecastInputScreen> {
  final _formKey = GlobalKey<FormState>();

  List<String> _crops = [];
  List<String> _markets = [];

  String? _crop;
  String? _market;

  int _quantityKg = 25;
  int _daysSinceHarvest = 5;
  int _daysAhead = 7;

  bool _loading = true;
  bool _submitting = false;
  String _err = "";

  // ✅ Web-like date UI (backend logic unchanged)
  DateTime _predictDate = DateTime.now();
  String _dataAvailableUpTo = "2026-02-16";

  @override
  void initState() {
    super.initState();
    _loadCrops();
  }

  Future<void> _loadCrops() async {
    setState(() {
      _loading = true;
      _err = "";
    });

    try {
      final data = await MarketApi.getCrops();
      final crops =
          (data["crops"] as List? ?? []).map((e) => e.toString()).toList();

      setState(() {
        _crops = crops;
        _crop = crops.isNotEmpty ? crops.first : null;
      });

      if (_crop != null) {
        await _loadMarkets(_crop!);
      }

      if (mounted) setState(() => _loading = false);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _err = "$e";
      });
    }
  }

  Future<void> _loadMarkets(String crop) async {
    try {
      final markets = await MarketApi.getMarkets(crop);
      if (!mounted) return;
      setState(() {
        _markets = markets;
        _market = markets.isNotEmpty ? markets.first : null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _markets = [];
        _market = null;
        _err = "$e";
      });
    }
  }

  void _generateForecast() async {
    FocusScope.of(context).unfocus();
    if (!_formKey.currentState!.validate()) return;
    if (_crop == null || _market == null) return;

    setState(() => _submitting = true);

    await Future.delayed(const Duration(milliseconds: 220)); // tiny UX delay

    if (!mounted) return;
    setState(() => _submitting = false);

    Navigator.of(context).push(
      MaterialPageRoute(
        builder: (_) => MarketForecastAnalyticsScreen(
          crop: _crop!,
          market: _market!,
          quantityKg: _quantityKg,
          daysSinceHarvest: _daysSinceHarvest,
          daysAhead: _daysAhead,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final canSubmit = !_loading &&
        !_submitting &&
        _crop != null &&
        _market != null &&
        _markets.isNotEmpty;

    return AppShell(
      currentIndex: 0,
      child: Column(
        children: [
          // ✅ MODERN HEADER (hero + small illustration area)
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
                  right: -14,
                  top: 22,
                  child: Opacity(
                    opacity: 0.10,
                    child: Icon(Icons.show_chart_rounded,
                        size: 150, color: Colors.white),
                  ),
                ),
                const Positioned(
                  right: 18,
                  bottom: -18,
                  child: Opacity(
                    opacity: 0.08,
                    child: Icon(Icons.storefront_rounded,
                        size: 130, color: Colors.white),
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
                          "Forecast",
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
                          child: const Icon(Icons.person_rounded,
                              color: Colors.white),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    Text(
                      "Price Forecast",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.98),
                        fontWeight: FontWeight.w900,
                        fontSize: 26,
                        height: 1.05,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      "Choose crop & market, set horizon and\nget a smart recommendation.",
                      style: TextStyle(
                        color: Colors.white.withOpacity(0.72),
                        fontWeight: FontWeight.w600,
                        fontSize: 12.8,
                        height: 1.25,
                      ),
                    ),
                    const SizedBox(height: 12),

                    // ✅ “free space” as a modern banner card inside header
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.symmetric(
                          horizontal: 12, vertical: 10),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.10),
                        borderRadius: BorderRadius.circular(18),
                        border:
                            Border.all(color: Colors.white.withOpacity(0.14)),
                      ),
                      child: Row(
                        children: [
                          Container(
                            width: 38,
                            height: 38,
                            decoration: BoxDecoration(
                              color: Colors.white.withOpacity(0.12),
                              borderRadius: BorderRadius.circular(14),
                              border: Border.all(
                                  color: Colors.white.withOpacity(0.14)),
                            ),
                            child: const Icon(Icons.lightbulb_rounded,
                                color: Colors.white),
                          ),
                          const SizedBox(width: 10),
                          Expanded(
                            child: Text(
                              "Tip: Use 30 days horizon for better trend smoothing.",
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.86),
                                fontWeight: FontWeight.w700,
                                fontSize: 12.2,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),

          const SizedBox(height: 12),

          Expanded(
            child: _loading
                ? const Center(child: CircularProgressIndicator())
                : ListView(
                    physics: const BouncingScrollPhysics(),
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 18),
                    children: [
                      if (_err.isNotEmpty) ...[
                        _errorCard(_err),
                        const SizedBox(height: 12),
                      ],

                      // ✅ Prediction Settings (dark web-like)
                      _cardDark(
                        title: "Prediction Settings",
                        subtitle: "Pick a date to generate forecast",
                        child: Column(
                          children: [
                            _infoRow(
                                "Data available up to", _dataAvailableUpTo),
                            const SizedBox(height: 10),
                            _dateField(),
                          ],
                        ),
                      ),

                      const SizedBox(height: 12),

                      // ✅ Main Input Form (modern sections)
                      Container(
                        padding: const EdgeInsets.all(16),
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
                        child: Form(
                          key: _formKey,
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              _sectionTitle(
                                icon: Icons.tune_rounded,
                                title: "Market Inputs",
                                desc: "These values affect the prediction.",
                              ),
                              const SizedBox(height: 14),
                              _label("Crop"),
                              const SizedBox(height: 6),
                              _dropdown(
                                value: _crop,
                                items: _crops,
                                onChanged: (v) async {
                                  if (v == null) return;
                                  setState(() {
                                    _crop = v;
                                    _market = null;
                                    _markets = [];
                                  });
                                  await _loadMarkets(v);
                                },
                              ),
                              const SizedBox(height: 14),
                              _label("Market"),
                              const SizedBox(height: 6),
                              _dropdown(
                                value: _market,
                                items: _markets,
                                onChanged: (v) => setState(() => _market = v),
                              ),
                              const SizedBox(height: 16),
                              _label("Forecast Horizon"),
                              const SizedBox(height: 8),
                              _horizonChips(),
                              const SizedBox(height: 18),
                              _label("Quantity (kg)"),
                              const SizedBox(height: 8),
                              _stepperCard(
                                value: _quantityKg,
                                hint: "How much you plan to sell",
                                onMinus: () => setState(() => _quantityKg =
                                    (_quantityKg - 1).clamp(1, 9999)),
                                onPlus: () => setState(() => _quantityKg =
                                    (_quantityKg + 1).clamp(1, 9999)),
                              ),
                              const SizedBox(height: 14),
                              _label("Days Since Harvest"),
                              const SizedBox(height: 8),
                              _stepperCard(
                                value: _daysSinceHarvest,
                                hint: "Freshness affects demand",
                                onMinus: () => setState(() =>
                                    _daysSinceHarvest =
                                        (_daysSinceHarvest - 1).clamp(0, 365)),
                                onPlus: () => setState(() => _daysSinceHarvest =
                                    (_daysSinceHarvest + 1).clamp(0, 365)),
                              ),
                              const SizedBox(height: 18),
                              SizedBox(
                                width: double.infinity,
                                height: 54,
                                child: ElevatedButton(
                                  onPressed:
                                      canSubmit ? _generateForecast : null,
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: AppColors.darkGreen,
                                    foregroundColor: Colors.white,
                                    disabledBackgroundColor:
                                        AppColors.darkGreen.withOpacity(0.45),
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(18),
                                    ),
                                    elevation: 0,
                                  ),
                                  child: _submitting
                                      ? const SizedBox(
                                          height: 18,
                                          width: 18,
                                          child: CircularProgressIndicator(
                                            strokeWidth: 2,
                                            color: Colors.white,
                                          ),
                                        )
                                      : const Text(
                                          "Generate Forecast",
                                          style: TextStyle(
                                            fontWeight: FontWeight.w900,
                                            fontSize: 15.5,
                                          ),
                                        ),
                                ),
                              ),
                              const SizedBox(height: 10),
                              _softNote(
                                icon: Icons.lock_clock_rounded,
                                text:
                                    "We use your inputs to create forecast analytics & recommendations.",
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
          ),
        ],
      ),
    );
  }

  // ----------------- UI helpers -----------------
  Widget _sectionTitle({
    required IconData icon,
    required String title,
    required String desc,
  }) {
    return Row(
      children: [
        Container(
          width: 38,
          height: 38,
          decoration: BoxDecoration(
            color: AppColors.primary.withOpacity(0.18),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppColors.border),
          ),
          child: Icon(icon, color: AppColors.darkGreen),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  fontWeight: FontWeight.w900,
                  color: AppColors.textDark,
                  fontSize: 14.5,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                desc,
                style: TextStyle(
                  fontWeight: FontWeight.w700,
                  color: AppColors.textDark.withOpacity(0.55),
                  fontSize: 12.2,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _label(String t) => Text(
        t,
        style: const TextStyle(
          fontWeight: FontWeight.w900,
          color: AppColors.textDark,
          fontSize: 13.2,
        ),
      );

  Widget _softNote({required IconData icon, required String text}) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Icon(icon, size: 18, color: AppColors.textDark.withOpacity(0.6)),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                color: AppColors.textDark.withOpacity(0.70),
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _errorCard(String msg) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.red.withOpacity(0.06),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.red.withOpacity(0.2)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline_rounded, color: Colors.red),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              msg,
              style: const TextStyle(
                color: Colors.red,
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ✅ better horizon selector (chips, not dropdown)
  Widget _horizonChips() {
    final options = const [7, 14, 30];

    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: options.map((d) {
          final sel = d == _daysAhead;
          return Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 4),
              child: InkWell(
                borderRadius: BorderRadius.circular(14),
                onTap: () => setState(() => _daysAhead = d),
                child: Container(
                  padding: const EdgeInsets.symmetric(vertical: 10),
                  decoration: BoxDecoration(
                    color: sel
                        ? AppColors.primary.withOpacity(0.30)
                        : Colors.transparent,
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(
                      color: sel
                          ? AppColors.primary.withOpacity(0.60)
                          : AppColors.border,
                    ),
                  ),
                  child: Center(
                    child: Text(
                      "${d}D",
                      style: TextStyle(
                        fontWeight: FontWeight.w900,
                        color: sel
                            ? AppColors.darkGreen
                            : AppColors.textDark.withOpacity(0.70),
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

  Widget _dropdown({
    required String? value,
    required List<String> items,
    required ValueChanged<String?> onChanged,
  }) {
    return DropdownButtonFormField<String>(
      value: value,
      onChanged: onChanged,
      items:
          items.map((e) => DropdownMenuItem(value: e, child: Text(e))).toList(),
      decoration: InputDecoration(
        filled: true,
        fillColor: AppColors.surface,
        contentPadding:
            const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: AppColors.border),
        ),
      ),
      validator: (v) => (v == null || v.isEmpty) ? "Required" : null,
    );
  }

  Widget _stepperCard({
    required int value,
    required String hint,
    required VoidCallback onMinus,
    required VoidCallback onPlus,
  }) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          _roundIconBtn(icon: Icons.remove_rounded, onTap: onMinus),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "$value",
                  style: const TextStyle(
                    fontWeight: FontWeight.w900,
                    fontSize: 18,
                    color: AppColors.textDark,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  hint,
                  style: TextStyle(
                    fontWeight: FontWeight.w700,
                    color: AppColors.textDark.withOpacity(0.55),
                    fontSize: 12.2,
                  ),
                ),
              ],
            ),
          ),
          _roundIconBtn(icon: Icons.add_rounded, onTap: onPlus),
        ],
      ),
    );
  }

  Widget _roundIconBtn({required IconData icon, required VoidCallback onTap}) {
    return InkWell(
      borderRadius: BorderRadius.circular(14),
      onTap: onTap,
      child: Container(
        width: 44,
        height: 44,
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.border),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.04),
              blurRadius: 12,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Icon(icon, color: AppColors.darkGreen),
      ),
    );
  }

  // -------- web-like prediction settings widgets --------
  Widget _cardDark({
    required String title,
    required String subtitle,
    required Widget child,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: const Color(0xFF1E242B),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.08)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: TextStyle(
              color: Colors.white.withOpacity(0.92),
              fontWeight: FontWeight.w900,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            subtitle,
            style: TextStyle(
              color: Colors.white.withOpacity(0.70),
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

  Widget _infoRow(String k, String v) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.06),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withOpacity(0.10)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              k,
              style: TextStyle(
                color: Colors.white.withOpacity(0.75),
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
          Text(
            v,
            style: TextStyle(
              color: Colors.white.withOpacity(0.95),
              fontWeight: FontWeight.w900,
            ),
          ),
        ],
      ),
    );
  }

  Widget _dateField() {
    final label = "${_predictDate.year.toString().padLeft(4, '0')}/"
        "${_predictDate.month.toString().padLeft(2, '0')}/"
        "${_predictDate.day.toString().padLeft(2, '0')}";

    return InkWell(
      onTap: () async {
        final picked = await showDatePicker(
          context: context,
          initialDate: _predictDate,
          firstDate: DateTime(2020, 1, 1),
          lastDate: DateTime(2035, 12, 31),
        );
        if (picked != null && mounted) setState(() => _predictDate = picked);
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
        decoration: BoxDecoration(
          color: const Color(0xFF2A3038),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: Colors.white.withOpacity(0.10)),
        ),
        child: Row(
          children: [
            const Icon(Icons.calendar_month_rounded, color: Colors.white70),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                label,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w900,
                ),
              ),
            ),
            Icon(Icons.keyboard_arrow_down_rounded,
                color: Colors.white.withOpacity(0.7)),
          ],
        ),
      ),
    );
  }
}
