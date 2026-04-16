import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Upload, Ruler, UserRound, Sparkles, Image as ImageIcon, Scale, CircleGauge, ShieldCheck } from "lucide-react";

export default function FrontendPrediccionMedidas() {
  const [genero, setGenero] = useState("femenino");
  const [altura, setAltura] = useState("");
  const [imagen, setImagen] = useState(null);
  const [preview, setPreview] = useState("");
  const [cargando, setCargando] = useState(false);
  const [error, setError] = useState("");
  const [resultado, setResultado] = useState(null);

  const API_URL = "https://backend-predicciones-543132885604.us-central1.run.app/predict";

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const medidaEsperada = useMemo(() => {
    return genero === "femenino"
      ? { etiqueta: "Perímetro de cadera", clave: "cadera_cm", unidad: "cm" }
      : { etiqueta: "Perímetro de abdomen", clave: "abdomen_cm", unidad: "cm" };
  }, [genero]);

  const handleImageChange = (e) => {
    const file = e.target.files?.[0] || null;

    if (preview) URL.revokeObjectURL(preview);

    setImagen(file);
    setResultado(null);
    setError("");

    if (file) {
      setPreview(URL.createObjectURL(file));
    } else {
      setPreview("");
    }
  };

  const validarFormulario = () => {
    if (!altura) return "Ingresa la altura en metros.";

    const alturaNum = Number(altura);
    if (Number.isNaN(alturaNum)) return "La altura debe ser un número válido.";
    if (alturaNum < 1.2 || alturaNum > 2.3) return "La altura debe estar entre 1.20 m y 2.30 m.";
    if (!imagen) return "Debes cargar una imagen de cuerpo completo.";

    return "";
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setResultado(null);

    const mensajeError = validarFormulario();
    if (mensajeError) {
      setError(mensajeError);
      return;
    }

    try {
      setCargando(true);

      const formData = new FormData();
      formData.append("gender", genero === "femenino" ? "female" : "male");
      formData.append("stature_m", altura);
      formData.append("file", imagen);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "No fue posible obtener una predicción.");
      }

      const data = await response.json();
      setResultado(data);
    } catch (err) {
      setError(err?.message || "Ocurrió un error al conectar con el backend.");
    } finally {
      setCargando(false);
    }
  };

  const predictions = resultado?.predictions || {};
  const peso = predictions.peso_estimado;
  const generoResp = resultado?.gender
    ? (resultado.gender === "female" ? "femenino" : "masculino")
    : genero;

  const cadera = generoResp === "femenino" ? predictions.circunferencia_estimada : null;
  const abdomen = generoResp === "masculino" ? predictions.circunferencia_estimada : null;
  const medidaResultado = generoResp === "femenino" ? cadera : abdomen;

  const cardsInfo = [
    {
      icon: UserRound,
      label: "Género",
      value: generoResp,
      format: (value) => <span className="capitalize">{value}</span>,
    },
    {
      icon: Scale,
      label: "Peso estimado",
      value: peso,
      format: (value) =>
        value !== undefined && value !== null && !Number.isNaN(Number(value))
          ? `${Number(value).toFixed(1)} kg`
          : "Sin dato",
    },
    {
      icon: CircleGauge,
      label: generoResp === "femenino" ? "Perímetro de cadera" : "Perímetro de abdomen",
      value: medidaResultado,
      format: (value) =>
        value !== undefined && value !== null && !Number.isNaN(Number(value))
          ? `${Number(value).toFixed(1)} metros`
          : "Sin dato",
    },
  ];

  return (
    <div className="min-h-screen bg-slate-950 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black p-4 text-slate-300 md:p-8 font-sans">
      <div className="mx-auto max-w-4xl">
        {/* Header Section */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-center mb-10"
        >
          <div className="inline-flex items-center gap-2 rounded-full border border-cyan-500/30 bg-cyan-500/10 px-4 py-1.5 text-xs font-semibold tracking-wide text-cyan-400 mb-6 shadow-[0_0_15px_rgba(6,182,212,0.15)]">
            <Sparkles className="h-4 w-4" />
            AI Body Metrics Scanner
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight text-white md:text-5xl lg:text-6xl bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-violet-500 pb-2">
            Predicción Anatómica
          </h1>
          <p className="mt-4 text-lg leading-relaxed text-slate-400 max-w-2xl mx-auto">
            Sube una foto frontal de cuerpo completo con fondo simple, ingresa tu estatura y la Inteligencia Artificial arrojará medidas precisas calculadas a partir de tus píxeles.
          </p>
        </motion.div>

        {/* Main Form Card */}
        <motion.form
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          onSubmit={handleSubmit}
          className="relative overflow-hidden rounded-[2rem] border border-slate-800 bg-slate-900/60 p-6 shadow-2xl backdrop-blur-xl md:p-10"
        >
          {/* Subtle glow border effect */}
          <div className="pointer-events-none absolute inset-0 rounded-[2rem] shadow-[inset_0_0_0_1px_rgba(255,255,255,0.05)]" />
          
          <div className="grid gap-8 md:grid-cols-2">
            {/* Left Column: Gender & Height */}
            <div className="space-y-8 flex flex-col justify-center">
              {/* Gender */}
              <div>
                <label className="mb-3 block text-sm font-semibold tracking-wide text-slate-300 uppercase">1. Biotipo de Perfil</label>
                <div className="flex gap-3">
                  {[
                    { value: "femenino", label: "Femenino", icon: UserRound },
                    { value: "masculino", label: "Masculino", icon: UserRound },
                  ].map((option) => (
                    <button
                      key={option.value}
                      type="button"
                      onClick={() => setGenero(option.value)}
                      className={`flex flex-1 items-center justify-center gap-2 rounded-2xl border px-4 py-3 text-sm font-medium transition-all ${
                        genero === option.value
                          ? "border-cyan-500 bg-cyan-500/10 text-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.2)]"
                          : "border-slate-800 bg-slate-950/50 text-slate-500 hover:border-slate-700 hover:text-slate-300"
                      }`}
                    >
                      <option.icon className="h-4 w-4" />
                      {option.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Height */}
              <div>
                <label className="mb-3 block text-sm font-semibold tracking-wide text-slate-300 uppercase">2. Estatura</label>
                <div className="relative group">
                  <div className="absolute inset-y-0 left-0 flex items-center pl-4 pointer-events-none">
                    <Ruler className="h-5 w-5 text-slate-500 group-focus-within:text-cyan-400 transition-colors" />
                  </div>
                  <input
                    type="number"
                    step="0.01"
                    min="1.20"
                    max="2.30"
                    placeholder="Metros (ej. 1.70)"
                    value={altura}
                    onChange={(e) => setAltura(e.target.value)}
                    className="w-full rounded-2xl border border-slate-800 bg-slate-950/50 py-4 pl-12 pr-4 text-white placeholder-slate-600 outline-none transition-all focus:border-cyan-500 focus:bg-slate-900 focus:shadow-[0_0_15px_rgba(6,182,212,0.15)]"
                  />
                </div>
              </div>
            </div>

            {/* Right Column: Upload/Preview Unified */}
            <div className="flex flex-col h-full">
              <label className="mb-3 block text-sm font-semibold tracking-wide text-slate-300 uppercase">3. Fotografía</label>
              
              <label className="relative flex flex-1 min-h-[240px] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-[1.5rem] border-2 border-dashed border-slate-700 bg-slate-950/30 p-2 text-center transition-all hover:border-violet-500 hover:bg-slate-900/50">
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/jpg"
                  onChange={handleImageChange}
                  className="hidden"
                />
                
                {preview ? (
                  <div className="group relative w-full h-full flex items-center justify-center bg-black/40 rounded-xl overflow-hidden">
                    <img
                      src={preview}
                      alt="Vista previa"
                      className="absolute inset-0 h-full w-full object-contain opacity-70 transition-opacity group-hover:opacity-20"
                    />
                    <div className="relative z-10 flex flex-col items-center opacity-0 transition-opacity group-hover:opacity-100 scale-95 group-hover:scale-100 duration-200">
                      <div className="rounded-full bg-slate-800/80 border border-slate-600 p-4 backdrop-blur-md shadow-lg">
                        <Upload className="h-7 w-7 text-white" />
                      </div>
                      <span className="mt-3 text-sm font-semibold text-white tracking-widest uppercase">Cambiar Archivo</span>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-6">
                    <div className="rounded-full border border-slate-800 bg-slate-900 p-5 shadow-inner mb-4 transition-transform hover:scale-105 hover:bg-slate-800">
                      <ImageIcon className="h-8 w-8 text-slate-400" />
                    </div>
                    <p className="text-sm font-medium text-slate-300">Haz clic o arrastra para cargar</p>
                    <p className="mt-1 text-xs text-slate-600 max-w-[200px]">Admite formatos JPG y PNG de alta resolución.</p>
                  </div>
                )}
              </label>
            </div>
          </div>

          {error && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="mt-6 rounded-2xl border border-red-900/50 bg-red-950/30 px-5 py-4 text-sm text-red-400 backdrop-blur-sm">
              <span className="font-semibold text-red-300">⚠️ Error:</span> {error}
            </motion.div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={cargando}
            className="mt-8 relative group w-full overflow-hidden rounded-[1.25rem] p-[2px] disabled:opacity-60 disabled:cursor-not-allowed"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 via-violet-500 to-fuchsia-500 opacity-70 group-hover:opacity-100 blur-sm transition-opacity duration-300"></div>
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-violet-600"></div>
            <div className="relative flex items-center justify-center gap-2 rounded-[18px] bg-slate-950/40 px-6 py-4 transition-all group-hover:bg-transparent">
              {cargando ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span className="text-base font-semibold text-white tracking-wide">Analizando Red Neuronal...</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-5 w-5 text-white" />
                  <span className="text-base font-bold text-white tracking-wide">Calcular Inteligencia Artificial</span>
                </>
              )}
            </div>
          </button>
        </motion.form>

        {/* Results Section */}
        {resultado && !error && (
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, type: "spring", bounce: 0.4 }}
            className="mt-12"
          >
            <div className="flex items-center justify-center gap-4 mb-8">
              <div className="h-px bg-gradient-to-r from-transparent to-slate-700 flex-1"></div>
              <h2 className="text-sm font-semibold tracking-[0.2em] text-slate-400 uppercase">Reporte de Inferencia</h2>
              <div className="h-px bg-gradient-to-l from-transparent to-slate-700 flex-1"></div>
            </div>

            <div className="grid gap-5 sm:grid-cols-3">
              {cardsInfo.map((item, index) => {
                const Icon = item.icon;
                return (
                  <motion.div
                    key={item.label}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.4, delay: 0.1 * index }}
                    className="relative overflow-hidden rounded-[2rem] border border-slate-800 bg-slate-900/80 p-6 flex flex-col items-center text-center shadow-[0_10px_40px_rgba(0,0,0,0.6)] backdrop-blur-xl group hover:border-violet-500/30 transition-colors"
                  >
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 w-1/2 h-[2px] bg-gradient-to-r from-transparent via-cyan-500 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
                    
                    <div className="mb-4 rounded-full bg-slate-950 p-4 text-cyan-400 shadow-inner group-hover:text-violet-400 transition-colors border border-slate-800 group-hover:border-slate-700">
                      <Icon className="h-7 w-7" />
                    </div>
                    <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest">{item.label}</p>
                    <div className="mt-3 text-3xl font-extrabold tracking-tight text-white drop-shadow-[0_0_10px_rgba(255,255,255,0.15)]">
                      {item.format(item.value)}
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
