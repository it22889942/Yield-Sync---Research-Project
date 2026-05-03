import 'package:flutter/material.dart';
import '../../utils/app_colors.dart';
import '../../utils/form_validators.dart';
import '../../services/labour_equipment_post_service.dart';
import '../../services/nav.dart';
import '../widgets/app_text_field.dart';
import '../widgets/post_availability_schedule.dart';
import '../widgets/primary_button.dart';

class AddLabourPostScreen extends StatefulWidget {
  const AddLabourPostScreen({super.key});

  @override
  State<AddLabourPostScreen> createState() => _AddLabourPostScreenState();
}

class _AddLabourPostScreenState extends State<AddLabourPostScreen> {
  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _hourlyRateCtrl = TextEditingController();
  final _experienceYearsCtrl = TextEditingController();

  bool _saving = false;
  String? _selectedLabourType;
  String? _selectedSkillLevel;
  String? _selectedDistrict;
  String? _selectedSeason;
  String? _selectedCropType;
  String? _selectedAvailability;
  TimeOfDay? _availableFrom;
  TimeOfDay? _availableTo;

  static const List<String> _labourTypes = [
    'Field Worker',
    'Rice Planting',
    'Rice Harvesting',
    'Irrigation',
    'Spraying',
    'Weeding',
    'Machine Operator',
  ];

  static const List<String> _skillLevels = [
    'Beginner',
    'Experienced',
  ];

  static const List<String> _districts = [
    'Alawwa',
    'Bingiriya',
    'Galgamuwa',
    'Giriulla',
    'Hettipola',
    'Hiriyala',
    'Ibbagamuwa',
    'Kuliyapitiya',
    'Kurunegala',
    'Maho',
    'Maspotha',
    'Mawathagama',
    'Narammala',
    'Nikaweratiya',
    'Pannala',
    'Polgahawela',
    'Rideegama',
    'Wariyapola',
    'Weerambugedara',
  ];

  static const List<String> _seasons = ['Yala', 'Maha'];

  static const List<String> _cropTypes = [
    'Paddy',
    'Maize',
    'Onion',
    'Chickpea',
    'Vegetables',
    'Other',
  ];

  static const List<String> _availabilityOptions = [
    'Weekdays',
    'Weekends',
    'Both',
  ];

  @override
  void dispose() {
    _nameCtrl.dispose();
    _hourlyRateCtrl.dispose();
    _experienceYearsCtrl.dispose();
    super.dispose();
  }

  String _formatTimeOfDay(TimeOfDay time) {
    final hour = time.hourOfPeriod == 0 ? 12 : time.hourOfPeriod;
    final minute = time.minute.toString().padLeft(2, '0');
    final suffix = time.period == DayPeriod.am ? 'AM' : 'PM';
    return '$hour:$minute $suffix';
  }

  String _formatLocation(String city) {
    if (city.toLowerCase() == 'kurunegala') return 'Kurunegala';
    return 'Kurunegala, $city';
  }

  Future<void> _pickFromTime() async {
    final selected = await showAppTimePicker(
      context,
      initialTime: _availableFrom ?? const TimeOfDay(hour: 8, minute: 0),
      helpText: 'Start time',
    );
    if (selected == null) return;
    setState(() => _availableFrom = selected);
  }

  Future<void> _pickToTime() async {
    final selected = await showAppTimePicker(
      context,
      initialTime: _availableTo ?? const TimeOfDay(hour: 17, minute: 0),
      helpText: 'End time',
    );
    if (selected == null) return;
    setState(() => _availableTo = selected);
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    if (_availableFrom == null || _availableTo == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Select available time range')),
      );
      return;
    }
    final fromMinutes = _availableFrom!.hour * 60 + _availableFrom!.minute;
    final toMinutes = _availableTo!.hour * 60 + _availableTo!.minute;
    if (toMinutes <= fromMinutes) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('"Available to" must be later than "Available from"')),
      );
      return;
    }

    setState(() => _saving = true);
    try {
      final hourlyRate = double.tryParse(_hourlyRateCtrl.text.trim()) ?? 0.0;
      final experienceYears = int.tryParse(_experienceYearsCtrl.text.trim()) ?? 0;

      await LabourEquipmentPostService.createLabourPost(
        name: _nameCtrl.text.trim(),
        labourType: _selectedLabourType!,
        skillLevel: _selectedSkillLevel!,
        location: _formatLocation(_selectedDistrict!),
        season: _selectedSeason!,
        cropType: _selectedCropType!,
        availableDay: _selectedAvailability!,
        availableTime: '${_formatTimeOfDay(_availableFrom!)} - ${_formatTimeOfDay(_availableTo!)}',
        hourlyRate: hourlyRate,
        experienceYears: experienceYears,
      );

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Labour post added. It will appear in the list.')),
      );
      Nav.back(context);
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed: $e')),
      );
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.surface,
      body: SafeArea(
        child: Column(
          children: [
            _header(),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 24),
                child: Form(
                  key: _formKey,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      _card(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Labour details',
                              style: TextStyle(
                                fontWeight: FontWeight.w900,
                                fontSize: 16,
                                color: AppColors.textDark,
                              ),
                            ),
                            const SizedBox(height: 14),
                            AppTextField(
                              controller: _nameCtrl,
                              label: 'Full name',
                              hint: 'Your name',
                              textCapitalization: TextCapitalization.words,
                              validator: (v) => FormValidators.requiredPersonName(
                                v,
                                fieldLabel: 'Full name',
                              ),
                            ),
                            const SizedBox(height: 12),
                            _dropdownField(
                              label: 'Labour type',
                              value: _selectedLabourType,
                              items: _labourTypes,
                              onChanged: (value) => setState(() => _selectedLabourType = value),
                              validatorMessage: 'Labour type required',
                            ),
                            const SizedBox(height: 12),
                            _dropdownField(
                              label: 'Skill level',
                              value: _selectedSkillLevel,
                              items: _skillLevels,
                              onChanged: (value) => setState(() => _selectedSkillLevel = value),
                              validatorMessage: 'Skill level required',
                            ),
                            const SizedBox(height: 12),
                            _dropdownField(
                              label: 'Location (Kurunegala)',
                              value: _selectedDistrict,
                              items: _districts,
                              onChanged: (value) => setState(() => _selectedDistrict = value),
                              validatorMessage: 'Location required',
                            ),
                            const SizedBox(height: 12),
                            _dropdownField(
                              label: 'Season',
                              value: _selectedSeason,
                              items: _seasons,
                              onChanged: (value) => setState(() => _selectedSeason = value),
                              validatorMessage: 'Season required',
                            ),
                            const SizedBox(height: 12),
                            _dropdownField(
                              label: 'Crop type',
                              value: _selectedCropType,
                              items: _cropTypes,
                              onChanged: (value) => setState(() => _selectedCropType = value),
                              validatorMessage: 'Crop type required',
                            ),
                            const SizedBox(height: 12),
                            _dropdownField(
                              label: 'Available days',
                              value: _selectedAvailability,
                              items: _availabilityOptions,
                              onChanged: (value) => setState(() => _selectedAvailability = value),
                              validatorMessage: 'Availability required',
                            ),
                            const SizedBox(height: 12),
                            PostAvailabilityTimeSection(
                              fromDisplay: _availableFrom == null
                                  ? null
                                  : _formatTimeOfDay(_availableFrom!),
                              toDisplay: _availableTo == null
                                  ? null
                                  : _formatTimeOfDay(_availableTo!),
                              onPickFrom: _pickFromTime,
                              onPickTo: _pickToTime,
                              onClearFrom: () =>
                                  setState(() => _availableFrom = null),
                              onClearTo: () =>
                                  setState(() => _availableTo = null),
                            ),
                            const SizedBox(height: 12),
                            AppTextField(
                              controller: _hourlyRateCtrl,
                              label: 'Hourly rate (LKR)',
                              hint: 'e.g. 500',
                              keyboardType:
                                  const TextInputType.numberWithOptions(
                                decimal: true,
                              ),
                              validator: (v) =>
                                  FormValidators.requiredPositiveNumber(
                                v,
                                'Hourly rate (LKR)',
                              ),
                            ),
                            const SizedBox(height: 12),
                            AppTextField(
                              controller: _experienceYearsCtrl,
                              label: 'Experience (years)',
                              hint: 'e.g. 5',
                              keyboardType:
                                  const TextInputType.numberWithOptions(
                                signed: false,
                                decimal: false,
                              ),
                              validator: (v) =>
                                  FormValidators.optionalWholeYears(
                                v,
                                'Experience (years)',
                              ),
                            ),
                            const SizedBox(height: 18),
                            PrimaryButton(
                              text: _saving ? 'Adding...' : 'Add labour post',
                              icon: _saving ? Icons.hourglass_top_rounded : Icons.person_add_rounded,
                              onPressed: _saving ? null : _submit,
                              isLoading: _saving,
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _header() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 18),
      decoration: const BoxDecoration(
        gradient: AppColors.heroGradient,
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(28),
          bottomRight: Radius.circular(28),
        ),
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Nav.back(context),
            icon: const Icon(Icons.arrow_back_ios_new_rounded),
            color: Colors.white,
          ),
          const SizedBox(width: 8),
          const Expanded(
            child: Text(
              'Add as labour',
              style: TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.w900,
                fontSize: 18,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _card({required Widget child}) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: AppColors.border),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 20,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: child,
    );
  }

  Widget _dropdownField({
    required String label,
    required String? value,
    required List<String> items,
    required ValueChanged<String?> onChanged,
    required String validatorMessage,
  }) {
    return DropdownButtonFormField<String>(
      value: value,
      isExpanded: true,
      icon: const Icon(Icons.keyboard_arrow_down_rounded, color: AppColors.darkGreen),
      borderRadius: BorderRadius.circular(14),
      dropdownColor: Colors.white,
      style: const TextStyle(
        color: AppColors.textDark,
        fontWeight: FontWeight.w600,
      ),
      decoration: _inputDecoration(label: label),
      hint: const Text(
        'Tap to open list',
        style: TextStyle(
          color: AppColors.muted,
          fontWeight: FontWeight.w600,
        ),
      ),
      items: items
          .map(
            (item) => DropdownMenuItem<String>(
              value: item,
              child: Text(
                item,
                style: const TextStyle(
                  color: AppColors.textDark,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          )
          .toList(),
      onChanged: onChanged,
      validator: (v) => v == null || v.isEmpty ? validatorMessage : null,
    );
  }

  InputDecoration _inputDecoration({
    required String label,
    Widget? suffixIcon,
  }) {
    const borderRadius = BorderRadius.all(Radius.circular(14));
    return InputDecoration(
      labelText: label,
      suffixIcon: suffixIcon,
      filled: true,
      fillColor: const Color(0xFFF7FCFA),
      contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
      labelStyle: const TextStyle(
        color: AppColors.muted,
        fontWeight: FontWeight.w600,
      ),
      enabledBorder: const OutlineInputBorder(
        borderRadius: borderRadius,
        borderSide: BorderSide(color: AppColors.border),
      ),
      focusedBorder: const OutlineInputBorder(
        borderRadius: borderRadius,
        borderSide: BorderSide(color: AppColors.primary, width: 1.6),
      ),
      errorBorder: const OutlineInputBorder(
        borderRadius: borderRadius,
        borderSide: BorderSide(color: AppColors.error),
      ),
      focusedErrorBorder: const OutlineInputBorder(
        borderRadius: borderRadius,
        borderSide: BorderSide(color: AppColors.error, width: 1.4),
      ),
    );
  }
}
