import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../widgets/app_shell.dart';
import '../../services/fertiliser_service.dart';
import '../../services/fertiliser_history_service.dart';

class FertilizerResultScreen extends StatefulWidget {
  final FertiliserPredictResult result;
  final String cropLabel;
  final String stageLabel;

  const FertilizerResultScreen({
    super.key,
    required this.result,
    required this.cropLabel,
    required this.stageLabel,
  });

  @override
  State<FertilizerResultScreen> createState() => _FertilizerResultScreenState();
}

class _FertilizerResultScreenState extends State<FertilizerResultScreen> {
  final _acreCtrl = TextEditingController();

  bool _loadingTotal = false;

  double? _totalYieldKg;
  double? _yieldKgPerAcreUsed;
  String? _source;

  // interactive area slider
  double _acreSlider = 1;

  @override
  void initState() {
    super.initState();
    _acreCtrl.text = "1";
    _acreSlider = 1;

    _acreCtrl.addListener(() {
      final v = double.tryParse(_acreCtrl.text.trim());
      if (v != null && v > 0) {
        final clamped = v.clamp(0.1, 100.0);
        if ((_acreSlider - clamped).abs() > 0.001) {
          setState(() => _acreSlider = clamped);
        }
      }
    });
  }

  @override
  void dispose() {
    _acreCtrl.dispose();
    super.dispose();
  }

  Future<void> _calculateTotalYield() async {
    final acres = double.tryParse(_acreCtrl.text.trim());
    if (acres == null || acres <= 0) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Enter a valid acre size")),
      );
      return;
    }

    final ypa = widget.result.yieldKgPerAcre;
    if (ypa == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Yield per acre not available.")),
      );
      return;
    }

    setState(() {
      _loadingTotal = true;
      _totalYieldKg = null;
      _yieldKgPerAcreUsed = null;
      _source = null;
    });

    try {
      final res = await FertiliserService.totalYield(
        areaAcres: acres,
        yieldKgPerAcre: ypa,
      );

      setState(() {
        _totalYieldKg = res.totalYieldKg;
        _yieldKgPerAcreUsed = res.yieldKgPerAcre;
        _source = res.source;
      });

      final fertType = widget.result.fertiliserType.isEmpty
          ? "—"
          : widget.result.fertiliserType;

      await FertiliserHistoryService().saveTotalYieldCalc(
        areaAcres: acres,
        yieldKgPerAcre: res.yieldKgPerAcre,
        totalYieldKg: res.totalYieldKg,
        source: res.source,
        fertiliserType: fertType,
        cropLabel: widget.cropLabel,
        stageLabel: widget.stageLabel,
      );

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Total Yield saved ✅")),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Calculate error: $e")),
      );
    } finally {
      if (mounted) setState(() => _loadingTotal = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final fertType = widget.result.fertiliserType.isEmpty
        ? "—"
        : widget.result.fertiliserType;

    final ypa = widget.result.yieldKgPerAcre;

    return AppShell(
      currentIndex: 0,
      child: Column(
        children: [
          // ===================== HERO HEADER =====================
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
            child: Column(
              children: [
                Row(
                  children: [
                    IconButton(
                      onPressed: () => Navigator.of(context).pop(),
                      icon: const Icon(Icons.arrow_back_ios_new_rounded),
                      color: Colors.white,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      "Fertilizer Result",
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
                      child: const Icon(Icons.verified_rounded,
                          color: Colors.white),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                Align(
                  alignment: Alignment.centerLeft,
                  child: Text(
                    "Prediction Summary",
                    style: TextStyle(
                      color: Colors.white.withOpacity(0.96),
                      fontWeight: FontWeight.w900,
                      fontSize: 22,
                    ),
                  ),
                ),
                const SizedBox(height: 10),
                Row(
                  children: [
                    Expanded(
                      child: _headerChip(
                        icon: Icons.spa_rounded,
                        label: widget.cropLabel,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: _headerChip(
                        icon: Icons.timeline_rounded,
                        label: widget.stageLabel,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                _headerChip(
                  icon: Icons.science_rounded,
                  label: "Fertilizer: $fertType",
                  full: true,
                ),
              ],
            ),
          ),

          const SizedBox(height: 12),

          // ===================== BODY =====================
          Expanded(
            child: ListView(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 110),
              children: [
                // ===== Highlight Yield Card =====
                _card(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _sectionTitle("PREDICTED YIELD"),
                      const SizedBox(height: 10),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(14),
                        decoration: BoxDecoration(
                          color: AppColors.primary.withOpacity(0.12),
                          borderRadius: BorderRadius.circular(18),
                          border: Border.all(color: AppColors.border),
                        ),
                        child: Row(
                          children: [
                            const Icon(Icons.auto_graph_rounded,
                                color: AppColors.darkGreen, size: 26),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Text(
                                ypa == null
                                    ? "—"
                                    : "${ypa.toStringAsFixed(2)} Kg / Acre",
                                style: const TextStyle(
                                  fontWeight: FontWeight.w900,
                                  fontSize: 18,
                                  color: AppColors.textDark,
                                ),
                              ),
                            ),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 10, vertical: 6),
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(999),
                                border: Border.all(color: AppColors.border),
                              ),
                              child: Text(
                                "Saved ✅",
                                style: TextStyle(
                                  fontWeight: FontWeight.w900,
                                  color: AppColors.textDark.withOpacity(0.7),
                                  fontSize: 12,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 12),
                      _kv("Crop", widget.cropLabel),
                      const SizedBox(height: 10),
                      _kv("Growth Stage", widget.stageLabel),
                      const SizedBox(height: 10),
                      _kv("Fertiliser Type", fertType),
                    ],
                  ),
                ),

                const SizedBox(height: 12),

                // ===== Field Size Card =====
                _card(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _sectionTitle("FIELD SIZE"),
                      const SizedBox(height: 10),

                      // slider + text sync
                      Row(
                        children: [
                          const Icon(Icons.crop_free_rounded,
                              color: AppColors.darkGreen),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              "Select your land area (acres)",
                              style: TextStyle(
                                fontWeight: FontWeight.w800,
                                color: AppColors.textDark.withOpacity(0.7),
                              ),
                            ),
                          ),
                          Container(
                            padding: const EdgeInsets.symmetric(
                                horizontal: 10, vertical: 6),
                            decoration: BoxDecoration(
                              color: AppColors.primary.withOpacity(0.12),
                              borderRadius: BorderRadius.circular(999),
                              border: Border.all(color: AppColors.border),
                            ),
                            child: Text(
                              "${_acreSlider.toStringAsFixed(1)} ac",
                              style: const TextStyle(
                                fontWeight: FontWeight.w900,
                                color: AppColors.darkGreen,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 10),
                      Slider(
                        value: _acreSlider.clamp(0.1, 100.0),
                        min: 0.1,
                        max: 100.0,
                        divisions: 999,
                        onChanged: (v) {
                          setState(() {
                            _acreSlider = v;
                            _acreCtrl.text = v.toStringAsFixed(1);
                          });
                        },
                      ),
                      const SizedBox(height: 6),
                      TextFormField(
                        controller: _acreCtrl,
                        keyboardType: const TextInputType.numberWithOptions(
                            decimal: true),
                        decoration: InputDecoration(
                          labelText: "Enter acres",
                          hintText: "e.g., 4",
                          filled: true,
                          fillColor: AppColors.surface,
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(16),
                            borderSide:
                                const BorderSide(color: AppColors.border),
                          ),
                          enabledBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(16),
                            borderSide:
                                const BorderSide(color: AppColors.border),
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(16),
                            borderSide: const BorderSide(
                                color: AppColors.primary, width: 1.6),
                          ),
                        ),
                      ),
                      const SizedBox(height: 10),
                      Text(
                        "Tip: Use the slider for quick input, or type exact value.",
                        style: TextStyle(
                          color: AppColors.textDark.withOpacity(0.55),
                          fontWeight: FontWeight.w700,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ),

                // ===== Final Result Card =====
                if (_totalYieldKg != null) ...[
                  const SizedBox(height: 12),
                  _card(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _sectionTitle("FINAL RESULT"),
                        const SizedBox(height: 10),
                        Container(
                          width: double.infinity,
                          padding: const EdgeInsets.all(14),
                          decoration: BoxDecoration(
                            color: Colors.green.withOpacity(0.08),
                            borderRadius: BorderRadius.circular(18),
                            border: Border.all(
                              color: Colors.green.withOpacity(0.22),
                            ),
                          ),
                          child: Row(
                            children: [
                              const Icon(Icons.check_circle_rounded,
                                  color: Colors.green, size: 28),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(
                                  "${_totalYieldKg!.toStringAsFixed(2)} Kg Total Yield",
                                  style: const TextStyle(
                                    fontWeight: FontWeight.w900,
                                    fontSize: 17,
                                    color: AppColors.textDark,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                        const SizedBox(height: 12),
                        _kv("Fertiliser Type", fertType),
                        const SizedBox(height: 10),
                        _kv(
                          "Yield Used (Kg/Acre)",
                          (_yieldKgPerAcreUsed ?? 0).toStringAsFixed(2),
                        ),
                        const SizedBox(height: 10),
                        _kv(
                          "Total Yield (Kg)",
                          _totalYieldKg!.toStringAsFixed(2),
                        ),
                        const SizedBox(height: 10),
                        Row(
                          children: [
                            _badge(
                              icon: Icons.storage_rounded,
                              text: "Saved ✅",
                            ),
                            const SizedBox(width: 10),
                            _badge(
                              icon: Icons.info_rounded,
                              text: "Source: ${_source ?? "provided"}",
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ],
            ),
          ),

          // ===================== STICKY CTA =====================
          SafeArea(
            top: false,
            child: Container(
              padding: const EdgeInsets.fromLTRB(16, 10, 16, 14),
              decoration: BoxDecoration(
                color: Colors.white,
                border: Border(
                  top: BorderSide(color: AppColors.border.withOpacity(0.9)),
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.06),
                    blurRadius: 18,
                    offset: const Offset(0, -10),
                  ),
                ],
              ),
              child: SizedBox(
                height: 52,
                width: double.infinity,
                child: ElevatedButton.icon(
                  onPressed: _loadingTotal ? null : _calculateTotalYield,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: AppColors.darkGreen,
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                  icon: _loadingTotal
                      ? const SizedBox(
                          width: 18,
                          height: 18,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: AppColors.darkGreen,
                          ),
                        )
                      : const Icon(Icons.calculate_rounded),
                  label: Text(
                    _loadingTotal ? "Calculating..." : "Calculate Total Yield",
                    style: const TextStyle(fontWeight: FontWeight.w900),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===================== UI HELPERS =====================

  Widget _headerChip(
      {required IconData icon, required String label, bool full = false}) {
    return Container(
      width: full ? double.infinity : null,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.12),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.16)),
      ),
      child: Row(
        children: [
          Icon(icon, size: 18, color: Colors.white.withOpacity(0.95)),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: TextStyle(
                color: Colors.white.withOpacity(0.92),
                fontWeight: FontWeight.w800,
                fontSize: 12.5,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _card({required Widget child}) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 18,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: child,
    );
  }

  Widget _sectionTitle(String title) {
    return Text(
      title,
      style: TextStyle(
        color: AppColors.textDark.withOpacity(0.55),
        fontWeight: FontWeight.w900,
        fontSize: 11.5,
        letterSpacing: 1.2,
      ),
    );
  }

  Widget _kv(String k, String v) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 12),
      decoration: BoxDecoration(
        color: AppColors.primary.withOpacity(0.10),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppColors.border),
      ),
      child: Row(
        children: [
          Expanded(
            child: Text(
              k,
              style: TextStyle(
                color: AppColors.textDark.withOpacity(0.75),
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

  Widget _badge({required IconData icon, required String text}) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 10),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppColors.border),
        ),
        child: Row(
          children: [
            Icon(icon, size: 18, color: AppColors.darkGreen),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                text,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: TextStyle(
                  fontWeight: FontWeight.w900,
                  color: AppColors.textDark.withOpacity(0.8),
                  fontSize: 12.5,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
