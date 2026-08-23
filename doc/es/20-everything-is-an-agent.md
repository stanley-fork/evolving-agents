# Todo es un agente

*Un rediseño del escritorio, del demo y del tour. Escrito antes del código,
porque la versión anterior estaba construida sobre un ejemplo que dejó de ser el
proyecto.*

---

## 0. La pregunta que esto responde

> NeXT hizo que todo en la UI fuera un objeto. Ahora todo es un agente. ¿Cómo se
> ve un escritorio donde todo está vivo, donde lo que se ve son los flujos de
> información, y donde podés inspeccionar cualquier cosa — o poner un agente de
> sistema a inspeccionarla por vos?

Ese es el encargo. El resto de este documento es lo que cuesta tomarlo
literalmente, y qué hay que tirar para hacerlo.

---

## 1. Lo que el escritorio anterior le mostraba a un desconocido

No es una crítica de gusto. Cuatro cosas verificables abriendo el archivo
publicado en `/demo/`.

**El scope de aterrizaje era ficción.** `renderDeskHtml` se llamaba con
`scopeId: "group:web-project-demo"`. Los documentos eran *Ledger currency
rewrite* y *Duplicate ledger rows*. No existe tal proyecto. Los dos proyectos que
sostienen todo el argumento — `coclea-sr`, cuyas 135 verificaciones corrieron en
verde en un runner de GitHub en 23m27s, y `hemo-verified`, cuyo panel de oráculos
está medido contra 98 filas de error conocido — eran, respectivamente, la cuarta
opción de un desplegable y *nada*.

La primera pantalla era un ejemplo inventado. Todo lo real estaba detrás de un
`<select>`.

**Los agentes eran mobiliario, no objetos.** La superficie tenía un `.shelf`
rotulado "Agents" contra un borde; los documentos eran las cosas puestas sobre el
escritorio y los cubos eran su decoración. Exactamente al revés del encargo.

**Los flujos de información no se veían.** Había una traza para leer, un digest
que resumía, tiras verdes. No había ningún lugar en la pantalla donde *ver algo
moverse de un agente a otro*. La palabra "flujo" aparecía; el flujo no.

**Nada inspeccionaba nada.** El riel tenía un panel llamado *Selected* que
imprimía datos de lo que hubieras clickeado. No había inspector en el sentido de
NeXT — un panel, atado a la selección, mostrando los campos reales del objeto — y
mucho menos un agente haciendo la inspección.

Y el tour: siete de sus trece beats narraban el proyecto web inventado. La
compuerta de la cóclea — un resultado retenido en `blocked` por un oráculo
declarado antes de correr, que es lo mejor que hay en el repositorio — no se
alcanzaba nunca.

---

## 2. El espíritu del canvas, releído

`doc/04-ai-ui.md` nombra cuatro propiedades: **Espacial, Vivo, Generado,
Dirigible**. Releídas contra el encargo, tres se sostienen intactas y una estaba
escrita demasiado chica.

*Espacial* se sostiene. *Vivo* se sostiene. *Generado* se sostiene — y
`doc/15-generated-interaction.md` fase 5, **especificada y nunca construida**,
resulta ser la que carga el peso:

> Arrastrá el cubo ReviewAgent sobre el cubo MigrationAgent. Eso *es* declararlo
> subagente. El modelo escribe el diff en `MigrationAgent.md`. El escritorio deja
> de ser un visor de ai-os y se vuelve un editor de ai-os.

Ese gesto — *arrastrar un agente sobre otra cosa declara una relación, y la
relación queda escrita* — es exactamente el movimiento del Interface Builder de
NeXT. Ahí arrastrabas un cable de un objeto a otro y la conexión se volvía real,
en el archivo. Nadie la tipeaba.

La propiedad escrita demasiado chica es **Dirigible**. El canvas dice que el
usuario puede actuar sobre lo que ve. Pero el encargo pide algo más fuerte, y es
justo lo que ai-os puede hacer y un diagrama no:

> **le podés poner un agente encima.**

No "podés inspeccionarlo" — podés *delegar el inspeccionar*. Un inspector que es
él mismo un agente, sujeto a las mismas reglas que los demás, dejando el mismo
rastro. Esa es una quinta propiedad, y es la que vale la pena tener.

**La metáfora también tiene que moverse.** El escritorio de System 7 era correcto
para "documentos dispuestos en el espacio". Es incorrecto para "información
moviéndose entre cosas vivas", porque en un escritorio de System 7 nada se mueve
salvo que vos lo muevas. La referencia es NeXT: la paleta de objetos, el canvas,
el **Inspector**, y sobre todo las **conexiones que podías ver e inspeccionar**.

---

## 3. La inversión

> El documento era el objeto y el agente su decoración.
> **Invertilo.** El agente es el objeto; el documento es el rastro que deja.

Concretamente, cuatro cosas cambian en la superficie.

### 3.1 Los agentes son los nodos

Fuera del estante, sobre el escritorio. Un agente es una caja con un nombre, un
estado y las herramientas que tiene permitido usar. Está vivo: está ocioso o está
trabajando, y cuál de las dos se ve sin clickear.

El estante no desaparece — se vuelve la **paleta**, que es lo que siempre fue en
IB: el lugar del que arrastrás objetos *nuevos*.

### 3.2 Los flujos son cables, y los cables llevan cosas que podés abrir

Un flow no es una lista de pasos en un panel. Es un **camino a través de los
agentes**, dibujado en el escritorio, con lo que se movió viajando por él.

La disciplina que impide que esto sea una animación:

> **Un cable lleva un artefacto real o no lleva nada.**

Si un salto produjo un archivo — un reporte de compuerta, una línea de ledger, un
puntaje de oráculo, un veredicto congelado — el paquete sobre ese cable tiene esa
dirección, y clickearlo abre los bytes. Si un salto no produjo nada registrado,
el cable se dibuja **unknown**: punteado, gris, rotulado. Nunca plausible, nunca
suavizado.

Eso es la separación entre `blockers` y `unknown` de `freezeVerdict`, hecha
visual. *No corrió* no es *pasó*, y no debe dibujarse igual.

### 3.3 Un Inspector, atado a la selección

NeXT tenía exactamente un panel inspector. Clickeabas otro objeto y el panel
pasaba a ser sobre ese objeto. Nunca había duda de qué panel mirar.

El escritorio recibe lo mismo: un panel, que inspecciona lo que esté
seleccionado — un agente, un cable, un paquete, una compuerta, un documento. Para
cada uno muestra los **campos reales**: de un agente, su markdown; de un paquete,
los bytes que se movieron; de una compuerta, el JSON del reporte con el número y
la tolerancia uno al lado del otro.

### 3.4 El Inspector tiene una segunda posición: ponerle un agente encima

El panel tiene un interruptor.

- **Leerlo** — los campos, como arriba. Mirás vos.
- **Preguntarle a un agente** — arrastrás `INSPECTOR` encima. Es un agente como
  cualquier otro: aparece en el escritorio, da un paso, cuesta algo, y produce un
  hallazgo.

Y la regla que hace que la segunda posición valga más que una ventana de chat:

> **Cada hallazgo cita el artefacto del que salió, y la cita se puede clickear.**

El agente de sistema dice "GATE-A01 reporta 2.592e-4 contra una tolerancia de
1.0e-4, así que esta cadena no puede congelar" — y al lado de esa frase está el
reporte que leyó. Estás a un click de verificarlo. Cuando no tiene artefacto que
citar, está obligado a decir `unknown`, y el escritorio dibuja eso distinto de
una respuesta.

Esta es la tesis entera del repositorio, expresada como una prestación de la
interfaz y no como un párrafo de un README: *el juicio de un modelo es una
afirmación; una afirmación necesita una dirección.*

---

## 4. Los dos proyectos, que son el demo

El proyecto web inventado deja de ser el scope de aterrizaje. El demo abre sobre
trabajo real, y los dos proyectos reales están elegidos porque **están en
desacuerdo sobre si la verdad es derivable**, que es lo más interesante que
cualquiera de los dos tiene para decir.

### coclea-sr — la verdad es derivable, así que dejá que la derive el código

Un modelo coclear cuyos autovalores tienen forma cerrada. `truth/` la computa y
**tiene prohibido importar `src/`**. Una compuerta es un oráculo declarado antes
de correr.

Dos cadenas, ocho agentes cada una, seis pasos cada una, **las dos enteramente en
verde**. Una está equivocada en cada número que reporta — un error de masa
`O(dx)` en el helicotrema, invisible para todos los chequeos de sanidad y de
hecho *preferido* por el ingenuo. No congela, porque GATE-A01 dice 2.592e-4
contra una tolerancia de 1.0e-4 y GATE-A12 dice que el orden de convergencia es
0.9996 donde debería ser 2.

El escritorio muestra: dos caminos de cables que se ven idénticos, uno terminando
en un resultado congelado, otro terminando en una compuerta roja que se puede
abrir.

### hemo-verified — la verdad no es derivable, así que medí al juez

Flujo sanguíneo, donde no hay forma cerrada para los casos que importan. Así que
armás un panel de siete oráculos y hacés lo que casi nadie hace: **medís el
panel** contra 98 filas cuyo error verdadero conocés.

Los números están en `gates/reports/h0.json`, y son humillantes a propósito:

| qué | valor |
|---|---|
| filas | 98 |
| aceptadas / rechazadas / escaladas | 48 / 32 / 18 |
| AUC compuesto | 0.9056 |
| oráculo individual más débil (A5) | 0.5209 |
| tasa de falso-acepta | 0.0208 |

Seis de los siete oráculos, solos, están cerca de tirar una moneda. El compuesto
no. **Ese es un hecho sobre juzgar que sólo se obtiene midiendo**, y está
registrado con el hash de cada oráculo y el entorno exacto — python 3.13.12,
numpy 2.5.2, scipy 1.18.1 — que lo produjo.

Puestos uno al lado del otro en el mismo escritorio, los dos proyectos dicen:
*cuando podés derivar la respuesta, derivala y que el código chequee; cuando no
podés, no sustituyas con un modelo confiado — medí qué tan bueno es realmente tu
juez, y publicá el número.*

Ese es el aporte, y por eso esto no es otro framework de agentes.

---

## 5. El tour, rehecho

La misma regla de antes, intacta y no negociable: **el tour maneja el cliente
real con eventos reales.** Si el escritorio se rompe, el tour se rompe. Nada acá
dibuja un cuadro ni anima un falso.

Lo que cambia es de qué se trata. Nueve beats:

1. **El escritorio son agentes.** No un diagrama — cada caja es algo con
   herramientas y un estado, y dos están trabajando ahora.
2. **Un flow es un camino.** Seguí un paquete de `DERIVADOR` a la compuerta.
3. **Abrí el paquete.** Estos son los bytes que se movieron. No un resumen.
4. **Dos cadenas, las dos verdes.** Seleccionalas lado a lado. Nada en ninguna
   traza las separa.
5. **La compuerta las separa.** Una congeló. La otra está retenida en `blocked`.
   Abrí GATE-A01: 2.592e-4 contra 1.0e-4.
6. **Ponele un agente encima.** Arrastrá `INSPECTOR` sobre la cadena retenida.
   Corre. Responde — y cita el reporte.
7. **Verificá la cita.** Un click. El número de la frase es el número del
   archivo. *Este es el beat por el que existe todo el tour.*
8. **El otro proyecto.** Cambiá a hemo. Acá no hay forma cerrada, así que el juez
   mismo está en juicio: 0.9056 compuesto, 0.5209 para A5 solo.
9. **Es tuyo.** Arrastrá lo que quieras.

El tour termina en el gesto y no en un resumen, porque el gesto es el argumento.

---

## 6. Qué mantiene esto honesto

Un rediseño que hiciera el escritorio más lindo y los números más vagos sería una
pérdida, así que el rediseño viene con un chequeo.

`scripts/check-demo-provenance.py` lee el demo construido y, para cada número que
muestra atribuido a un artefacto de un proyecto, resuelve ese artefacto y
compara. El desacuerdo rompe el build. El demo no puede desviarse de los
proyectos ni los proyectos del demo — la misma regla que `test/cochlea-demo.test.ts`
ya impone para los autovalores, aplicada a todo lo que el rediseño pone en
pantalla.

Las reglas existentes siguen: el demo es **generado** por el código del propio
producto, nunca mantenido a mano; la simulación se inyecta sólo bajo `simulate`;
la página no simulada no contiene tour.

---

## 7. Lo que esto no hace

Dicho sin vueltas, porque un documento de diseño que sólo enumera victorias es
publicidad.

- **No corre el cronómetro.** `NEXT.md` dice "no más escritorio antes del
  cronómetro" — la falsación de M5, un usuario y un flow de tres días que no
  corrió él, cronometrado. Este trabajo está explícitamente autorizado a pasar
  por encima de esa regla, y no la salda. El rediseño debería hacer el cronómetro
  *más fácil* de correr — un visitante que puede seguir un paquete y abrirlo es
  exactamente la medición que M5 quiere — pero hasta que se corra, ai-ui sigue
  sin falsar.
- **No hace del agente inspector un modelo real.** En el demo publicado no hay
  modelo; el hallazgo lo produce la misma simulación que produce todo lo demás, y
  está rotulado como tal en el chrome. Lo que sí es real es la *cita*: el
  artefacto que señala es el artefacto del repositorio, y el chequeo de
  procedencia lo demuestra.
- **No borra los otros scopes.** El laboratorio de señal y el de memoria cargan
  cada uno una falsación que los dos proyectos no tienen — un paso que corrió y
  no llevó nada, y un flow verde y equivocado. Dejan de ser el aterrizaje y se
  quedan en la paleta.
