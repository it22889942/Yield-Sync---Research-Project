/// Shared [FormField] / [TextFormField] validators for post flows.
class FormValidators {
  FormValidators._();

  /// True if [s] contains at least one letter (Latin or Sinhala block).
  static bool _hasLetter(String s) {
    return RegExp(r'[A-Za-z\u0D80-\u0DFF]').hasMatch(s);
  }

  /// Required person name: not empty, not digits-only, must contain letters.
  static String? requiredPersonName(String? v, {String fieldLabel = 'Name'}) {
    final t = v?.trim() ?? '';
    if (t.isEmpty) return '$fieldLabel is required';
    if (RegExp(r'^\d+$').hasMatch(t)) {
      return '$fieldLabel cannot be numbers only';
    }
    if (!_hasLetter(t)) {
      return '$fieldLabel must contain letters';
    }
    return null;
  }

  /// Same as [requiredPersonName] but allows empty (optional field).
  static String? optionalPersonName(String? v, {String fieldLabel = 'Name'}) {
    final t = v?.trim() ?? '';
    if (t.isEmpty) return null;
    if (RegExp(r'^\d+$').hasMatch(t)) {
      return '$fieldLabel cannot be numbers only';
    }
    if (!_hasLetter(t)) {
      return '$fieldLabel must contain letters';
    }
    return null;
  }

  /// Mobile: optional; if filled, must be exactly 10 digits (non-digits stripped).
  static String? optionalPhone10Digits(String? v) {
    final raw = v?.trim() ?? '';
    if (raw.isEmpty) return null;
    final digits = raw.replaceAll(RegExp(r'\D'), '');
    if (digits.length != 10) {
      return 'Mobile number must be exactly 10 digits';
    }
    return null;
  }

  /// Required positive decimal (rates).
  static String? requiredPositiveNumber(String? v, String fieldLabel) {
    final t = v?.trim() ?? '';
    if (t.isEmpty) return '$fieldLabel is required';
    final n = double.tryParse(t);
    if (n == null) return '$fieldLabel must be a valid number';
    if (n <= 0) return '$fieldLabel must be greater than 0';
    return null;
  }

  /// If non-empty, must be a positive number (for optional rate fields).
  static String? optionalPositiveNumber(String? v, String fieldLabel) {
    final t = v?.trim() ?? '';
    if (t.isEmpty) return null;
    final n = double.tryParse(t);
    if (n == null) return '$fieldLabel must be a valid number';
    if (n <= 0) return '$fieldLabel must be greater than 0';
    return null;
  }

  /// Experience years: optional; if filled, whole number ≥ 0.
  static String? optionalWholeYears(String? v, String fieldLabel) {
    final t = v?.trim() ?? '';
    if (t.isEmpty) return null;
    final n = int.tryParse(t);
    if (n == null) return '$fieldLabel must be a whole number';
    if (n < 0) return '$fieldLabel cannot be negative';
    return null;
  }
}
