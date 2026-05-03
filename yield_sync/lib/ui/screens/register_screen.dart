import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../utils/app_colors.dart';
import '../../services/nav.dart';
import '../../services/auth_service.dart';
import '../widgets/app_text_field.dart';

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final _formKey = GlobalKey<FormState>();

  final _firstCtrl = TextEditingController();
  final _lastCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _userCtrl = TextEditingController();
  final _phoneCtrl = TextEditingController();
  final _passCtrl = TextEditingController();

  // ✅ dataset IDs
  final _labourIdCtrl = TextEditingController(); // L0001
  final _equipmentIdCtrl = TextEditingController(); // E0001 (Equipment_ID)

  bool _obscure = true;

  // ✅ types now
  String _userType = "farmer"; // farmer | labour | equipment

  bool _loading = false;

  @override
  void dispose() {
    _firstCtrl.dispose();
    _lastCtrl.dispose();
    _emailCtrl.dispose();
    _userCtrl.dispose();
    _phoneCtrl.dispose();
    _passCtrl.dispose();
    _labourIdCtrl.dispose();
    _equipmentIdCtrl.dispose();
    super.dispose();
  }

  String? _req(String? v, String msg) {
    if (v == null || v.trim().isEmpty) return msg;
    return null;
  }

  Future<void> _register() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _loading = true);

    try {
      final auth = AuthService();

      if (_userType == "farmer") {
        // ✅ keep old registerUser to avoid affecting other parts
        await auth.registerUser(
          firstName: _firstCtrl.text,
          lastName: _lastCtrl.text,
          username: _userCtrl.text,
          phone: _phoneCtrl.text,
          email: _emailCtrl.text,
          password: _passCtrl.text,
          userType: "farmer",
        );
      } else if (_userType == "labour") {
        await auth.registerLabour(
          labourId: _labourIdCtrl.text,
          firstName: _firstCtrl.text,
          lastName: _lastCtrl.text,
          username: _userCtrl.text,
          phone: _phoneCtrl.text,
          email: _emailCtrl.text,
          password: _passCtrl.text,
        );
      } else if (_userType == "equipment") {
        await auth.registerEquipment(
          equipmentId: _equipmentIdCtrl.text,
          firstName: _firstCtrl.text,
          lastName: _lastCtrl.text,
          username: _userCtrl.text,
          phone: _phoneCtrl.text,
          email: _emailCtrl.text,
          password: _passCtrl.text,
        );
      }

      if (!mounted) return;

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text("Account created ✅ Please login.")),
      );

      Nav.back(context);
    } on FirebaseAuthException catch (e) {
      final msg = switch (e.code) {
        'email-already-in-use' => "Email already registered.",
        'invalid-email' => "Invalid email address.",
        'weak-password' => "Weak password (min 6 characters).",
        _ => e.message ?? "Registration failed.",
      };

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(msg)));
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text("Error: $e")),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final isLabour = _userType == "labour";
    final isEquipment = _userType == "equipment";

    return Scaffold(
      backgroundColor: const Color(0xFFF9FDF2),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(18, 18, 18, 18),
          children: [
            Row(
              children: [
                IconButton(
                  onPressed: () => Nav.back(context),
                  icon: const Icon(Icons.arrow_back_ios_new_rounded),
                ),
                const SizedBox(width: 6),
                const Text(
                  "Register",
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerLeft,
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: AppColors.primary.withOpacity(0.18),
                  borderRadius: BorderRadius.circular(999),
                  border: Border.all(color: AppColors.border),
                ),
                child: Text(
                  "CREATE ACCOUNT",
                  style: GoogleFonts.inter(
                    color: AppColors.darkGreen,
                    fontSize: 10.5,
                    letterSpacing: 1.0,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 14),
            Container(
              padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: AppColors.border),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "Join",
                          style: GoogleFonts.poppins(
                            fontSize: 29,
                            fontWeight: FontWeight.w500,
                            color: AppColors.textDark,
                            height: 1.0,
                          ),
                        ),
                        Text(
                          "YieldSync",
                          style: GoogleFonts.poppins(
                            fontSize: 29,
                            fontWeight: FontWeight.w800,
                            color: AppColors.textDark,
                            height: 1.0,
                          ),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          "Set up your profile and start using smart tools.",
                          style: GoogleFonts.inter(
                            color: AppColors.textDark.withOpacity(0.68),
                            fontSize: 13.2,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 10),
                  Container(
                    width: 74,
                    height: 74,
                    decoration: BoxDecoration(
                      color: AppColors.primary.withOpacity(0.22),
                      borderRadius: BorderRadius.circular(22),
                      border: Border.all(color: AppColors.border),
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(10),
                      child: Image.asset(
                        "assets/images/logo.png",
                        fit: BoxFit.contain,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.fromLTRB(16, 18, 16, 16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(22),
                    border: Border.all(color: AppColors.border),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withOpacity(0.08),
                        blurRadius: 26,
                        offset: const Offset(0, 16),
                      ),
                    ],
                  ),
                  child: Form(
                    key: _formKey,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "Personal Information",
                          style: GoogleFonts.poppins(
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                            color: AppColors.textDark,
                          ),
                        ),
                        const SizedBox(height: 12),

                        Row(
                          children: [
                            Expanded(
                              child: AppTextField(
                                controller: _firstCtrl,
                                label: "First Name",
                                hint: "Kalana",
                                validator: (v) =>
                                    _req(v, "First name is required"),
                                suffix: const Icon(Icons.badge_outlined),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: AppTextField(
                                controller: _lastCtrl,
                                label: "Last Name",
                                hint: "Kasun",
                                validator: (v) =>
                                    _req(v, "Last name is required"),
                                suffix: const Icon(Icons.badge_outlined),
                              ),
                            ),
                          ],
                        ),

                        const SizedBox(height: 12),

                        AppTextField(
                          controller: _emailCtrl,
                          label: "Email",
                          hint: "name@gmail.com",
                          keyboardType: TextInputType.emailAddress,
                          validator: (v) {
                            final r = _req(v, "Email is required");
                            if (r != null) return r;
                            final ok = RegExp(r'^[^@]+@[^@]+\.[^@]+$')
                                .hasMatch(v!.trim());
                            if (!ok) return "Enter a valid email";
                            return null;
                          },
                          suffix: const Icon(Icons.email_outlined),
                        ),

                        const SizedBox(height: 12),

                        AppTextField(
                          controller: _userCtrl,
                          label: "Username",
                          hint: "username",
                          validator: (v) {
                            final r = _req(v, "Username is required");
                            if (r != null) return r;
                            if (v!.trim().length < 3) {
                              return "Minimum 3 characters";
                            }
                            return null;
                          },
                          suffix: const Icon(Icons.person_outline_rounded),
                        ),

                        const SizedBox(height: 12),

                        AppTextField(
                          controller: _phoneCtrl,
                          label: "Contact Number",
                          hint: "07XXXXXXXX",
                          keyboardType: TextInputType.phone,
                          validator: (v) {
                            final r = _req(v, "Contact number is required");
                            if (r != null) return r;
                            final digitsOnly = v!.replaceAll(RegExp(r'\D'), '');
                            if (digitsOnly.length < 9) {
                              return "Enter a valid number";
                            }
                            return null;
                          },
                          suffix: const Icon(Icons.phone_outlined),
                        ),

                        const SizedBox(height: 12),

                        const Text(
                          "User Type",
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w800,
                            color: AppColors.textDark,
                          ),
                        ),
                        const SizedBox(height: 6),

                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 12),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(14),
                            border: Border.all(color: AppColors.border),
                          ),
                          child: DropdownButtonFormField<String>(
                            value: _userType,
                            decoration: const InputDecoration(
                              border: InputBorder.none,
                            ),
                            items: const [
                              DropdownMenuItem(
                                value: "farmer",
                                child: Text("Farmer (Normal)"),
                              ),
                              DropdownMenuItem(
                                value: "labour",
                                child: Text("Labour ID"),
                              ),
                              DropdownMenuItem(
                                value: "equipment",
                                child: Text("Equipment Owner ID"),
                              ),
                            ],
                            onChanged: (v) {
                              if (v == null) return;
                              setState(() => _userType = v);
                            },
                          ),
                        ),

                        // ✅ Labour ID
                        if (isLabour) ...[
                          const SizedBox(height: 12),
                          AppTextField(
                            controller: _labourIdCtrl,
                            label: "Labour ID",
                            hint: "L0001",
                            validator: (v) => _req(v, "Labour ID is required"),
                            suffix: const Icon(Icons.badge_rounded),
                          ),
                          Text(
                            "Must match dataset Labour_ID (ex: L0001).",
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                              color: AppColors.textDark.withOpacity(0.55),
                            ),
                          ),
                        ],

                        // ✅ Equipment ID
                        if (isEquipment) ...[
                          const SizedBox(height: 12),
                          AppTextField(
                            controller: _equipmentIdCtrl,
                            label: "Equipment ID",
                            hint: "E0001",
                            validator: (v) =>
                                _req(v, "Equipment ID is required"),
                            suffix: const Icon(Icons.badge_rounded),
                          ),
                          Text(
                            "Must match dataset Equipment_ID (ex: E0001).",
                            style: TextStyle(
                              fontSize: 12,
                              fontWeight: FontWeight.w700,
                              color: AppColors.textDark.withOpacity(0.55),
                            ),
                          ),
                        ],

                        const SizedBox(height: 12),

                        AppTextField(
                          controller: _passCtrl,
                          label: "Password",
                          hint: "Create a password",
                          obscure: _obscure,
                          validator: (v) {
                            final r = _req(v, "Password is required");
                            if (r != null) return r;
                            if (v!.length < 6) return "Minimum 6 characters";
                            return null;
                          },
                          suffix: IconButton(
                            onPressed: () =>
                                setState(() => _obscure = !_obscure),
                            icon: Icon(_obscure
                                ? Icons.visibility_off
                                : Icons.visibility),
                          ),
                        ),

                        const SizedBox(height: 16),

                        SizedBox(
                          width: double.infinity,
                          height: 52,
                          child: ElevatedButton.icon(
                            onPressed: _loading ? null : _register,
                            icon:
                                const Icon(Icons.person_add_alt_rounded, size: 18),
                            label: Text(
                              _loading ? "Creating..." : "Create Account",
                              style: GoogleFonts.poppins(
                                fontWeight: FontWeight.w700,
                                letterSpacing: 0.2,
                              ),
                            ),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppColors.primary,
                              foregroundColor: AppColors.darkGreen,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(16),
                              ),
                            ),
                          ),
                        ),

                        const SizedBox(height: 14),

                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(
                              "Already have an account?",
                              style: TextStyle(
                                color: AppColors.textDark.withOpacity(0.75),
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            TextButton(
                              onPressed: () => Nav.back(context),
                              child: const Text(
                                "Login",
                                style: TextStyle(fontWeight: FontWeight.w900),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
            const SizedBox(height: 18),
          ],
        ),
      ),
    );
  }
}
