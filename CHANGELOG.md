# CHANGELOG


## [0.13.0](https://github.com/pmcdi/jarvais/compare/v0.12.1...v0.13.0) (2025-05-21)


### Features

* add categorical mapping into outlier module. ([#77](https://github.com/pmcdi/jarvais/issues/77)) ([44c3915](https://github.com/pmcdi/jarvais/commit/44c39158c13d76969c5e195ab65cfe88c592e126))

## [0.12.1](https://github.com/pmcdi/jarvais/compare/v0.12.0...v0.12.1) (2025-05-20)


### Miscellaneous Chores

* release 0.12.1 ([03c4daa](https://github.com/pmcdi/jarvais/commit/03c4daa5e9280257564e2c23205746c086ce9e9a))
* release 0.12.1 ([818a4ba](https://github.com/pmcdi/jarvais/commit/818a4ba06f5034126a14d0d13c28a59ac900140f))

## [0.12.0](https://github.com/pmcdi/jarvais/compare/v0.11.3...v0.12.0) (2025-05-14)


### Features

* add new logger ([52aedd8](https://github.com/pmcdi/jarvais/commit/52aedd84e66a31f97774666ed0195d3b1838a800))
* new logger ([6f23529](https://github.com/pmcdi/jarvais/commit/6f235299a66980c991edb1d2dd9bd5fada6c1321))

## [0.11.3](https://github.com/pmcdi/jarvais/compare/v0.11.2...v0.11.3) (2025-04-25)


### Bug Fixes

* **analyzer:** fix minor bugs in Analyzer class ([952c00e](https://github.com/pmcdi/jarvais/commit/952c00ec6922ce520fb9f50e2f0b631822ee8aa9))
* remove side affect in plot_pairplot by using .copy ([5749ae0](https://github.com/pmcdi/jarvais/commit/5749ae0b88989b9b36c2551addf3b5ac885d44a0))
* slicing when applying mapping ([de18c1e](https://github.com/pmcdi/jarvais/commit/de18c1e42e612594995548cd8a7fc09f0a17892f))

## [0.11.2](https://github.com/pmcdi/jarvais/compare/v0.11.1...v0.11.2) (2025-04-24)


### Bug Fixes

* autogluon model dep issue + reorg ([eea6666](https://github.com/pmcdi/jarvais/commit/eea6666deecece83b81d57136f3a9d11faa65a79))
* **dependencies:** autogluon import on some models ([7c39af1](https://github.com/pmcdi/jarvais/commit/7c39af19c5819f88c5e5afc5406cc9fc20b22da6))
* upgrade autogluon to 1.2 for all models ([a19e78d](https://github.com/pmcdi/jarvais/commit/a19e78dca11e3746eaa966f336f3c40a76590669))

## v0.11.1 (2025-02-19)

### Bug Fixes

- Cicd issues. Updating to latest pixi version
  ([`5deadb8`](https://github.com/pmcdi/jarvais/commit/5deadb8fe10fd72ae610e0abe7c584e94ee1c70f))

- For replacing missing, adds filler as a category before adding to df
  ([`1ec4c6e`](https://github.com/pmcdi/jarvais/commit/1ec4c6ebc93d0813de53c3e9e1490367654b364f))

- Trying cicd without lock file
  ([`be94bf4`](https://github.com/pmcdi/jarvais/commit/be94bf4b9b8df169ce1ccc8fcba265723f0f8e6b))

### Build System

- Update pixi.lock
  ([`4485301`](https://github.com/pmcdi/jarvais/commit/44853018a1669d8a7b0591b77917b0e71583077d))

- Update pixi.lock
  ([`2f2cb42`](https://github.com/pmcdi/jarvais/commit/2f2cb42ca6eb00f5bff1438ebb8b4c2463221d85))

### Continuous Integration

- Added depandabot and updated pixi setep version
  ([`8b1d2b2`](https://github.com/pmcdi/jarvais/commit/8b1d2b24bd3d50730904dfb7ea0d5829eaccc9c8))

- Pinning to stable pixi versions
  ([`7a6a27b`](https://github.com/pmcdi/jarvais/commit/7a6a27b3f459e3b77044846c7135e27088482cba))

- Updated pixi versions to 0.40.2
  ([`7e45d99`](https://github.com/pmcdi/jarvais/commit/7e45d99cbe078e877f7978710aa0387bafd6a249))

- Updated ruff action to be locked: false
  ([`b31e73e`](https://github.com/pmcdi/jarvais/commit/b31e73e3553b3f8cd819391fec1285c9f2d3f318))


## v0.11.0 (2025-02-10)

### Build System

- Update lock
  ([`0ee0dce`](https://github.com/pmcdi/jarvais/commit/0ee0dce8bf7b6b4a782321da0839c0cc3ec466ec))

- **pypi**: Added documentation link to toml
  ([`c96f4a1`](https://github.com/pmcdi/jarvais/commit/c96f4a1b94c5cbe366a730096c6f19804d4817d2))

### Features

- Trigger patch release with dummy update
  ([`98ce02e`](https://github.com/pmcdi/jarvais/commit/98ce02eae92d9f01793dc370d3fb21b32deff837))


## v0.10.1 (2025-02-07)

### Bug Fixes

- Run semantic and pypi only if tests passed or skipped
  ([`7ce8395`](https://github.com/pmcdi/jarvais/commit/7ce839582b81d1bf57302af4c27ae77c442ad0fe))

- **ci**: Revert to running tests on all files
  ([`ee706bb`](https://github.com/pmcdi/jarvais/commit/ee706bb77f2424a4b279226effcfb510ad5025ee))

### Build System

- Update lock file
  ([`6e8e8e0`](https://github.com/pmcdi/jarvais/commit/6e8e8e0874cac607cbe3770b426e36bed04f6e0e))

### Documentation

- Update to readme and front page of docs
  ([`ca275e1`](https://github.com/pmcdi/jarvais/commit/ca275e1ce98ba66543ae3b6edc6a546cf6e59a94))


## v0.10.0 (2025-02-07)

### Bug Fixes

- Ci ([`7044759`](https://github.com/pmcdi/jarvais/commit/70447590f0d39c3914f9c2d0ebf46b1f55ce7a81))

- Ignore tests for docs/worflow and update to correct token
  ([`d71b29b`](https://github.com/pmcdi/jarvais/commit/d71b29bc967d36560f7f4b12a457d2848c5ac485))

- Pillow max image loading size
  ([`8df45fd`](https://github.com/pmcdi/jarvais/commit/8df45fdbdd26126859a5ec7b5a26544bc4e86721))

- Pytest stuff
  ([`ac161d1`](https://github.com/pmcdi/jarvais/commit/ac161d1c6adf1ef21c305b88c5bde6d9d0d8948c))

- Toml file split for relesase and hatch
  ([`831f903`](https://github.com/pmcdi/jarvais/commit/831f903f955901a57e2ee787caa48ccfe8a244d5))

- **cd**: Use consistent pixi version
  ([`e3627d6`](https://github.com/pmcdi/jarvais/commit/e3627d6c2dc69a26480b9ef6e89ce5d65fb3009a))

Signed-off-by: Joshua Siraj <72413722+JoshuaSiraj@users.noreply.github.com>

### Build System

- Pypi init changes
  ([`8fb7e84`](https://github.com/pmcdi/jarvais/commit/8fb7e843f7ed3ae122228942ec72376a8bc90a69))

- Update lock
  ([`610a581`](https://github.com/pmcdi/jarvais/commit/610a581f7b602cf39af1c8e8169a17ce92922eef))

- Update pixi
  ([`8c93b36`](https://github.com/pmcdi/jarvais/commit/8c93b3679f239d0c0ff3bda4faa32bb1a100d268))

- Update pixi lock
  ([`725050b`](https://github.com/pmcdi/jarvais/commit/725050bc448d1c2d69a9ea87e8fab2516fcf0926))

- Update to pixi lock/toml
  ([`9f68a4b`](https://github.com/pmcdi/jarvais/commit/9f68a4bd60277e9b6987ba77fbce008db66bd391))

- Updated lock
  ([`29efa71`](https://github.com/pmcdi/jarvais/commit/29efa7142f1595a8bb072215296febddb9920532))

### Continuous Integration

- Brought back ubuntu
  ([`7a66128`](https://github.com/pmcdi/jarvais/commit/7a66128c42072baddf47005ea4f9e618f63d9195))

- Testing windows
  ([`9cbd628`](https://github.com/pmcdi/jarvais/commit/9cbd6282dd6bd1e4d82eec52259bae3eba663022))

### Documentation

- Fix notebooks and Radcure intro
  ([`eda46ff`](https://github.com/pmcdi/jarvais/commit/eda46ff64b0b4498874dcfa6eb873b592b6495d7))

### Features

- Minor tweaks to pdf generation and partially refactor of pdf functions
  ([`6195425`](https://github.com/pmcdi/jarvais/commit/61954256ae8ac39eb2f3ebf963db3070c1d690fc))

- Semantic + pypi
  ([`7f01da5`](https://github.com/pmcdi/jarvais/commit/7f01da52b3b1fff75b1764ed050ffb8710118af3))

- Update pixi.toml
  ([`7635329`](https://github.com/pmcdi/jarvais/commit/7635329ac63e5cba9324f5dc137d9d6cb6297bcf))

Signed-off-by: Joshua Siraj <72413722+JoshuaSiraj@users.noreply.github.com>

- Update pixi.toml
  ([`c33fb17`](https://github.com/pmcdi/jarvais/commit/c33fb17c1a5fb940ff4c298494962dd8a1195dd1))

Signed-off-by: Joshua Siraj <72413722+JoshuaSiraj@users.noreply.github.com>

- Updated outlier analysis with better readability. all _add functions will be refactored next
  commit
  ([`b66a19d`](https://github.com/pmcdi/jarvais/commit/b66a19d446689314388f92c43dc5c29ceb8d4623))

- Working analyzer PDF revamp
  ([`2be7f7d`](https://github.com/pmcdi/jarvais/commit/2be7f7ddea1bbfabc94000b3ff7881ebcc8e5140))

### Refactoring

- Pdf changes
  ([`464d190`](https://github.com/pmcdi/jarvais/commit/464d1906e49738a70c3220013a439434de64d84e))

- Removed fancy_grid
  ([`0c3e63d`](https://github.com/pmcdi/jarvais/commit/0c3e63d05909ae28a98f78d35863b6c2b9b818ba))

- Removed print statements
  ([`e2b3493`](https://github.com/pmcdi/jarvais/commit/e2b349350f9a4fc90fe386717391024113387685))


## v0.9.99 (2025-02-05)

### Bug Fixes

- Changed naming of fit kwargs
  ([`7551956`](https://github.com/pmcdi/jarvais/commit/7551956ff6c223091c6034548c27b6780062e477))

- Ci error
  ([`e429870`](https://github.com/pmcdi/jarvais/commit/e42987009cd12404b1ac5eccdcc1ca0b37974444))

- Cleaned up outdated modules
  ([`982c7be`](https://github.com/pmcdi/jarvais/commit/982c7bee1fe2526c0985ddf20d7705f7f094b35f))

- Debug print statement removed
  ([`c60c5d5`](https://github.com/pmcdi/jarvais/commit/c60c5d51fdf37043bd62ba0b89ee96a26af0d108))

- Deep survival models predict function
  ([`174f68d`](https://github.com/pmcdi/jarvais/commit/174f68d5415553a541a3a65432ee32ee0bf68627))

- Dep errors
  ([`a559f8c`](https://github.com/pmcdi/jarvais/commit/a559f8c41fe0ad6648e143e8c80273e030a40a5f))

- Fstring mismatch
  ([`8185c05`](https://github.com/pmcdi/jarvais/commit/8185c055fff2f3e8b64390d71a48fb8143a3227a))

- Loading pkl survival models
  ([`37a5def`](https://github.com/pmcdi/jarvais/commit/37a5defd623bdc41abf340f6cc3c3dce160ce04e))

- Loading was failing because the hyperparams were not saved
  ([`efea4dc`](https://github.com/pmcdi/jarvais/commit/efea4dc7739498e9c5bf15779b21fcb8d04c6658))

- Loading was failing because the hyperparams were not saved
  ([`0467d20`](https://github.com/pmcdi/jarvais/commit/0467d205e026ac31e52959d0c789b59581f285d7))

- Missmathed recursive strings
  ([`1d4bfec`](https://github.com/pmcdi/jarvais/commit/1d4bfecad4f963460fff1aed279ead65089a885b))

- More refactoring + moved fonts into `utils` folder
  ([`9588270`](https://github.com/pmcdi/jarvais/commit/9588270ede62d68d1f1d8e210934791b9c581d73))

- Moved lightning log saving to output dir
  ([`25b2b6f`](https://github.com/pmcdi/jarvais/commit/25b2b6f69cd146dfc956b758991c85f11ac711b2))

- Removed normalization of labels and added norm in predict
  ([`e7e4c1e`](https://github.com/pmcdi/jarvais/commit/e7e4c1eb3b73b3a3c6e031841315bb162bc57791))

- Trainer loading and inference errors
  ([`9ad516f`](https://github.com/pmcdi/jarvais/commit/9ad516f63aa50ff087e8eb75392df4100d105c29))

- Update to arguments
  ([`2813fd8`](https://github.com/pmcdi/jarvais/commit/2813fd83d7f07589ff8bb68a0b170e4852dc9f8a))

- Updated api for bias explainer with jarvais
  ([`d548509`](https://github.com/pmcdi/jarvais/commit/d5485092b296490e37d6c340e7f6d135945eb5d6))

### Build System

- Added build docs
  ([`373b208`](https://github.com/pmcdi/jarvais/commit/373b208cd6bf7e3391cb99cead89de5d3d905588))

- Limiting floating points of bias metrics for "prettier" table structure
  ([`fb6f61e`](https://github.com/pmcdi/jarvais/commit/fb6f61e7974dc01e6e61a76928ddb43e99d53865))

- Same file as main
  ([`c8a21b6`](https://github.com/pmcdi/jarvais/commit/c8a21b63b58506a9f86bc43c6559ec14fd93afa3))

- Test
  ([`96b143e`](https://github.com/pmcdi/jarvais/commit/96b143e96dd2c661d8752259bb6959683d5cb2dc))

- Testing proper git config
  ([`13a3443`](https://github.com/pmcdi/jarvais/commit/13a3443c139dd791e8eba8ff6f59a412a5931d92))

- Update .gitignore to remove macOS and local backup files
  ([`d6d13c1`](https://github.com/pmcdi/jarvais/commit/d6d13c12492cfef4c2590ed78dbbe798e8d13587))

- Update pixi.lock
  ([`f6ac193`](https://github.com/pmcdi/jarvais/commit/f6ac193ddd63de583dd364931cf2811e7d636749))

- Update pixi.lock
  ([`6e49b97`](https://github.com/pmcdi/jarvais/commit/6e49b979d3a7b961ea375dcad5b8c738dd791404))

- Update to .toml
  ([`d8fa134`](https://github.com/pmcdi/jarvais/commit/d8fa134ff8dc720313c6c54d170f2e0a8871cd85))

- Update to lock and .toml
  ([`8500fdc`](https://github.com/pmcdi/jarvais/commit/8500fdcc14adfd1a4c124d67120020b6a3192556))

- Update to lock and .toml
  ([`30be516`](https://github.com/pmcdi/jarvais/commit/30be516cc5a6c8b9694da83d2c251b4da4d5d1ed))

- Update to lock and .toml
  ([`84ca363`](https://github.com/pmcdi/jarvais/commit/84ca363974cb596600d67d2e742d7a654b8cf946))

- Update to pixi.lock
  ([`576b809`](https://github.com/pmcdi/jarvais/commit/576b8097f9df6bb85e2823462e736d5c3faa5f7e))

- Update to pixi.lock again
  ([`12a3b12`](https://github.com/pmcdi/jarvais/commit/12a3b12f3b9083229a522ef65b87b82dd5480ae5))

- Updated dependencies
  ([`7d0ea6d`](https://github.com/pmcdi/jarvais/commit/7d0ea6d2d19831f27c41e4d6ac92e8d4c07bedaa))

- Updated packages
  ([`bafc423`](https://github.com/pmcdi/jarvais/commit/bafc423fa50741d4effce500fb796a92453a81cf))

### Chores

- A2r outputs
  ([`926e54d`](https://github.com/pmcdi/jarvais/commit/926e54ddeca378eaa73ed56c3112e81d8f94a072))

- Pr comments
  ([`40e22e1`](https://github.com/pmcdi/jarvais/commit/40e22e15bd6443789fbe197f5dbc4c4a8e626f1a))

- Update to output saving and example notebook
  ([`f780010`](https://github.com/pmcdi/jarvais/commit/f7800106c50538eb6190528144b696f81645d1ad))

### Code Style

- Bias output
  ([`cd40da9`](https://github.com/pmcdi/jarvais/commit/cd40da93264171f09051c013358455b26a8668da))

- Cleaning plot and trainer
  ([`4d8e000`](https://github.com/pmcdi/jarvais/commit/4d8e000a9d393820dbc45ab35feac9d3af589ff0))

- Consistency edits in analyzer
  ([`f3d9862`](https://github.com/pmcdi/jarvais/commit/f3d986277f7c08e61b93160f4733308612b799d1))

- Linting edits
  ([`d612b28`](https://github.com/pmcdi/jarvais/commit/d612b28e4f15efb1772949ec1f353500a3414487))

- Linting edits
  ([`e6981ac`](https://github.com/pmcdi/jarvais/commit/e6981ac03dadaca646e7bbe557538d9f2c2b8ded))

- More linting
  ([`1a0225b`](https://github.com/pmcdi/jarvais/commit/1a0225b84485c8d7a1e5f4ecb4c46b63014355b5))

- Update to explainer report
  ([`934c688`](https://github.com/pmcdi/jarvais/commit/934c68849fe3eb99199f51a17e4e6499b3c8c70f))

### Continuous Integration

- Added macos and py310
  ([`24866bb`](https://github.com/pmcdi/jarvais/commit/24866bb75da88b289187fe9044dfae3e833b7758))

- Fix to repeated line
  ([`c1b0fb4`](https://github.com/pmcdi/jarvais/commit/c1b0fb47fc77e81f53e49058cb21c19adeaa5dbe))

- Increased timeout
  ([`92290dc`](https://github.com/pmcdi/jarvais/commit/92290dca60d7490f21ba074b5b05ec9a9f669cfa))

- Increased timeout
  ([`001d4bb`](https://github.com/pmcdi/jarvais/commit/001d4bb902bbfaa3bc900ec5d6147309fdfe0550))

- Removed macos
  ([`f597260`](https://github.com/pmcdi/jarvais/commit/f5972602f946b1e034c72682319e393e40f402ca))

- Removed repeated code in toml
  ([`584b244`](https://github.com/pmcdi/jarvais/commit/584b244c03625abb44d76c39a6a346a19bf9cde4))

- Set locked to false
  ([`dafacd5`](https://github.com/pmcdi/jarvais/commit/dafacd553f65b6b8b7e766bdb706a584ba022651))

- Trying new thing for ci
  ([`2a0d52d`](https://github.com/pmcdi/jarvais/commit/2a0d52d12a117ee88bbf4eeae2fdeced0fc6645b))

- Update to pixi version
  ([`1bbce4b`](https://github.com/pmcdi/jarvais/commit/1bbce4b99b264345af4c716c3eeeea68f64ff9b9))

### Documentation

- Added bias explainer
  ([`f78f47f`](https://github.com/pmcdi/jarvais/commit/f78f47f2e8ded337c461cae6c0d9b291ec7634f6))

- Added example to plot_corr
  ([`3a9de16`](https://github.com/pmcdi/jarvais/commit/3a9de16956ebe94e51ccbe59a7bcac7c7fa4a6a9))

- Added example to plot_corr
  ([`87312b2`](https://github.com/pmcdi/jarvais/commit/87312b256f5f0d5a28feb55fd2bdefaba6f61bde))

- Added functional and pdf utils modules to api
  ([`3a94917`](https://github.com/pmcdi/jarvais/commit/3a94917f84cda0b1d4927fe01718d9a3030a8834))

- Added kaplan meier to analyzer docs
  ([`933e8ce`](https://github.com/pmcdi/jarvais/commit/933e8ce183b4a9a4d1d7f69eed8d9b1ad3d53288))

- Added pdf previews of analysis and explainer reports
  ([`7b9736d`](https://github.com/pmcdi/jarvais/commit/7b9736d68edc2490036c255643152b5aaa6d8f90))

- Added search to mkdocs.yml
  ([`edfcd24`](https://github.com/pmcdi/jarvais/commit/edfcd24446b2bafd32a9c54e19195c1b48b0d83f))

- Added search to mkdocs.yml
  ([`3604675`](https://github.com/pmcdi/jarvais/commit/36046751c175a92180ee34c4f8158b80de79d8da))

- Changed to jarvais
  ([`ed6bddf`](https://github.com/pmcdi/jarvais/commit/ed6bddf686e0d7c8651579dba8f279b6f7afd0f6))

- Intial docs
  ([`b3e7368`](https://github.com/pmcdi/jarvais/commit/b3e7368e9603a64c45646f35d91d6aa4d1699763))

- Misc changes
  ([`ff7694a`](https://github.com/pmcdi/jarvais/commit/ff7694aec573d1c722fbeb454b52f2c3298cb342))

- Misc type fixes
  ([`71abb48`](https://github.com/pmcdi/jarvais/commit/71abb485c153d29eaca4401fe3e4efc9720d4384))

- New api page for plotting functions
  ([`b30c028`](https://github.com/pmcdi/jarvais/commit/b30c028e9b4b0e26d72a580143b9bf8f14aabd60))

- Update to plotting functions doc strings + reorg
  ([`e814d4f`](https://github.com/pmcdi/jarvais/commit/e814d4fa3399910d0f8bcec7e4fb5afcc82ed916))

- Updated Analyzer and Trainer docstrings from 'Args' -> 'Attributes'
  ([`f5b7dce`](https://github.com/pmcdi/jarvais/commit/f5b7dce49e271bf5e3334960b2d9e3e3e3e3f810))

- Updated api settings for better toc nesting
  ([`969d1b4`](https://github.com/pmcdi/jarvais/commit/969d1b4533a7ea51391402b5df76e0f2a625a8b3))

- Updated explainer page to have figures organized by taks
  ([`6da9e50`](https://github.com/pmcdi/jarvais/commit/6da9e5008647cbee31cf0e0048dc440c64cb9f92))

- Updated intro page bash block
  ([`3c50afa`](https://github.com/pmcdi/jarvais/commit/3c50afafae7e4be8aa0a8ddf8d0d2dbd8dd37309))

- Updated notebook
  ([`9257127`](https://github.com/pmcdi/jarvais/commit/9257127a4af5a1fe91bb735d038ddc50c3e07329))

- Updated notebook
  ([`86124cc`](https://github.com/pmcdi/jarvais/commit/86124cc76b5d41186df5e9bffce419d680fafe4f))

- Updated tutorial notebooks for regression and classification
  ([`a27a39a`](https://github.com/pmcdi/jarvais/commit/a27a39a6524ed3dda6264d7e3d8ea84e08b9ed80))

### Features

- Added back bias to explainer
  ([`32577c3`](https://github.com/pmcdi/jarvais/commit/32577c3a7d8e0a15663a16db942d72f527018bc2))

- Added bias compatibility for survival
  ([`fc1bfbd`](https://github.com/pmcdi/jarvais/commit/fc1bfbd9f90a1521efe4d615554adb89df8311ae))

- Added bootstraped metrics plot for survival
  ([`28094af`](https://github.com/pmcdi/jarvais/commit/28094af865ee2502a82ee9a299601f2e752fcc0f))

- Added checkmark and x to bias table
  ([`7785e69`](https://github.com/pmcdi/jarvais/commit/7785e698fc39e47686710adc39f2012dd5ff039e))

- Added deep survival models
  ([`d824f90`](https://github.com/pmcdi/jarvais/commit/d824f90aa1d69ac2938cce14b9544cf21357dd7c))

- Added deep survival models from pycox
  ([`cc3f4d9`](https://github.com/pmcdi/jarvais/commit/cc3f4d9088aa2ba53f2a4f5249003a373cc5f0cb))

- Added new inter font
  ([`4361014`](https://github.com/pmcdi/jarvais/commit/4361014b27f9d7f7150efe6d7979f39a9db2e9c4))

- Added train to explainer plots
  ([`e95fdca`](https://github.com/pmcdi/jarvais/commit/e95fdcab7e4ef1da9a0b2f2cbf781eb4a827ecd1))

- Added violin plot of bootstrapped metrics and added regression model to main leaderboard.
  ([`d45c677`](https://github.com/pmcdi/jarvais/commit/d45c677e9f59d0a8e0c22fb92484bdef2bb6f252))

- Bagging as an option + default behaviour when k_folds=1
  ([`92fe0a8`](https://github.com/pmcdi/jarvais/commit/92fe0a8ef421165172d0ef0ebf952d03b9a7bb0e))

- Bias analyses on only happens for a sens feat if f_pval < 0.5
  ([`f999e2b`](https://github.com/pmcdi/jarvais/commit/f999e2b60f8b58da8abd90c24e709aafc5735084))

- Changed one-hot encoding separator to "|" for better clarity
  ([`ebfba88`](https://github.com/pmcdi/jarvais/commit/ebfba88a4f17fbaa98f547221f95019fe98bce38))

- Changed to root_mse
  ([`4107dce`](https://github.com/pmcdi/jarvais/commit/4107dce68893d51c2a86eb3ee25666dce313304a))

- Confidence interval for roc and precision
  ([`eaa0c4c`](https://github.com/pmcdi/jarvais/commit/eaa0c4cf0545d46ba0f0ce1d09a30c3485136555))

- Equal opportunity
  ([`1d05af1`](https://github.com/pmcdi/jarvais/commit/1d05af1bdd8a4b743f6845ac4a622d3e2d767e20))

- Equal opportunity (TPR)
  ([`81f3ed6`](https://github.com/pmcdi/jarvais/commit/81f3ed6919d3cc7be1b0d91d92b2a484e7f03e8d))

- Feature importance for time_to_event in explainer
  ([`d51607e`](https://github.com/pmcdi/jarvais/commit/d51607ed39567d56b48b8ac9ddcbcfc215ff8671))

- First push with preliminary tests and ci/cd
  ([`2d90f88`](https://github.com/pmcdi/jarvais/commit/2d90f888e0a6cccea270f4448881301df16cb8f8))

- Frequency tables for categorical columns
  ([`7c09def`](https://github.com/pmcdi/jarvais/commit/7c09def59875a1a811af72e87a06527ffa47ad25))

- Kaplan mier curves for time-to-event data in analyzer
  ([`87839e7`](https://github.com/pmcdi/jarvais/commit/87839e768e780a6003b9842728d23b0bcc843404))

- Leaderboard table changes to show extra metrics for train/val
  ([`45c3726`](https://github.com/pmcdi/jarvais/commit/45c3726f9df093bed49e888e12c3d17516e73763))

- Loading pretrained, inference func, explainer tests, keyword args for predictor, validation
  evaluation plot
  ([`e244882`](https://github.com/pmcdi/jarvais/commit/e244882306dd3cfe58f1d3a195dd9e092e512c77))

- Log rank test for kaplan meier
  ([`11e33a7`](https://github.com/pmcdi/jarvais/commit/11e33a7802923837ec08ef258a0a088ac0a4617b))

- Log rank test with violin plot and regression model per sens feat
  ([`c9e8bfa`](https://github.com/pmcdi/jarvais/commit/c9e8bfa94278487251b558cf1ffa8f7aecac135d))

- Mit LICENSE
  ([`4ed8a2e`](https://github.com/pmcdi/jarvais/commit/4ed8a2e13a858af69edde47d67016fdb4f21e429))

- One hot encode option in analyzer
  ([`b8db199`](https://github.com/pmcdi/jarvais/commit/b8db19965e7aea29c98f89e8d58fe67231b56b6a))

- One hot encode option in analyzer
  ([`58b35f1`](https://github.com/pmcdi/jarvais/commit/58b35f154c402aa40a53076ae554434d1b3ca0a0))

- Pdf report for explainer
  ([`4ef68da`](https://github.com/pmcdi/jarvais/commit/4ef68da4fc8955ccb281f78eadb8fae21b7670a8))

- Put warnings into log
  ([`1d5871c`](https://github.com/pmcdi/jarvais/commit/1d5871c600d8d3a0c08f98893d188edb0b242d96))

- Removed save_data arg. Data now always saved
  ([`2acfaba`](https://github.com/pmcdi/jarvais/commit/2acfabac28291f7ccfcd29f2b17d2f4e104e2d5d))

- Removed save_data arg. Data now always saved
  ([`d7328ed`](https://github.com/pmcdi/jarvais/commit/d7328ed11641c99b52c50b4fb2d788cedbcf3eb3))

- Removed saving of config from analyzer dryrun
  ([`cb87eeb`](https://github.com/pmcdi/jarvais/commit/cb87eebf4b84832a293e9447ebb1e3825a82ffef))

- Run bias analysis on all sens feats
  ([`ccf9695`](https://github.com/pmcdi/jarvais/commit/ccf969592fe73385c9cefa3decdf84b4a5b99fba))

- Save validation data used for deep survival model training
  ([`acf3f26`](https://github.com/pmcdi/jarvais/commit/acf3f267e763fa3e3b91ba587c7b49881142bd95))

- Save validation data used for deep survival model training
  ([`30c530e`](https://github.com/pmcdi/jarvais/commit/30c530e3617d91b70944fe9c55d96ca4a9eadfd1))

- Saving bootstrapped performance plots as 3 seperate
  ([`4073be3`](https://github.com/pmcdi/jarvais/commit/4073be3343bd695f6dc8167ee04a661ce79c3ab7))

- Temp way to save model summary
  ([`77bb678`](https://github.com/pmcdi/jarvais/commit/77bb678f0e373ae855d7b5acb25849f75c1f901c))

- Time_to_event support in trainer
  ([`63dbc1c`](https://github.com/pmcdi/jarvais/commit/63dbc1cfef81dff5b04c9af49055007d07d07db3))

- Trainer config for loading and info
  ([`c9892b3`](https://github.com/pmcdi/jarvais/commit/c9892b322fde46f59dc585135d6455a319748779))

- Trainer config for loading and info
  ([`3e803e3`](https://github.com/pmcdi/jarvais/commit/3e803e364d65d114dca0a1231cb6e8d9b5322875))

- Updaitng BiasExplainer to properly label different fairness metrics
  ([`ceb605c`](https://github.com/pmcdi/jarvais/commit/ceb605cff9208b8b36ff1feeebd58d18dc0cee3f))

- Update to docs
  ([`0b0b234`](https://github.com/pmcdi/jarvais/commit/0b0b234783f7ea34e20bf2eb7d1fca5c7d1ae0d2))

- Update to README
  ([`dda75c6`](https://github.com/pmcdi/jarvais/commit/dda75c644d408b73349a0b0cd56fc0ec71beec9c))

- Updated analyzer dry_run to display tableone
  ([`3bfa136`](https://github.com/pmcdi/jarvais/commit/3bfa1360e8a431672d0c918f897411c21f9da9e8))

- Updated analyzer outputs for clarity + outlier analysis and feature types always printed
  regardless of if config is present
  ([`c7352c6`](https://github.com/pmcdi/jarvais/commit/c7352c64a1a672a3a80f756acde17a366a74e69d))

- Updated bias to plot log rank and fit OLS
  ([`325f807`](https://github.com/pmcdi/jarvais/commit/325f807579d94eea031b231d3a661722e5090cb7))

- Updated bias to plot log rank and fit OLS
  ([`1fae0db`](https://github.com/pmcdi/jarvais/commit/1fae0dbd12851bbbc9c14281df4813ae7bc60804))

- Updated bias to plot log rank and fit OLS
  ([`8131e65`](https://github.com/pmcdi/jarvais/commit/8131e652f24e83f1db61382cf4a3179c720c889c))

- Updated bias to use mse for bias auditing regression predictions
  ([`be6394e`](https://github.com/pmcdi/jarvais/commit/be6394e3c78d6f2171d4dbdbdb02cdd0b6354cb0))

- Updated deep models to output risk
  ([`aab9a85`](https://github.com/pmcdi/jarvais/commit/aab9a85e5345f2e017d34acd16ba51b2e57ba8e9))

- Updated feature importance for survival models to only include Cox for now
  ([`5511493`](https://github.com/pmcdi/jarvais/commit/55114936f51b847274dcfad3bf97c0deb12a2059))

- Updated leader for mean and intervals of kfolds
  ([`ab3739b`](https://github.com/pmcdi/jarvais/commit/ab3739bc12d3a120f6ff6180566c02d06be5a09b))

- Updated trainer to infer and load from survival models
  ([`fa50c19`](https://github.com/pmcdi/jarvais/commit/fa50c192a23459cf7aab9b1f4a17bef4f723c9dc))

- Updated trainer to now load in full info from outputdir
  ([`3ee16dd`](https://github.com/pmcdi/jarvais/commit/3ee16ddc20caf44d3e68f8b04d7581b544ee339d))

- Updated trainer to now load in full info from outputdir
  ([`360d4af`](https://github.com/pmcdi/jarvais/commit/360d4af81d0425733ba3cb6a23d86f1fa262b1f0))

- Working SHAP outputs and major refactoring
  ([`c694e4f`](https://github.com/pmcdi/jarvais/commit/c694e4f41626c854bf748b269a9111f587fab2c3))

### Refactoring

- Analyzer module updated
  ([`cb12f2b`](https://github.com/pmcdi/jarvais/commit/cb12f2b622da36a175e0e3eeec54c8cd33435a12))

- Change to jarvais
  ([`35d1e1e`](https://github.com/pmcdi/jarvais/commit/35d1e1e530247f9a51da37ee0e2423958e3fcf51))

- Changed time_to_event to survival
  ([`3b8151b`](https://github.com/pmcdi/jarvais/commit/3b8151ba35e6434b1810d9e915c8f70eae869379))

- First iteration of moves
  ([`9ad1717`](https://github.com/pmcdi/jarvais/commit/9ad17175f324f5a378885d34c1d81efdba2dec7f))

- Kfold training moved to functional
  ([`da3b436`](https://github.com/pmcdi/jarvais/commit/da3b43668580aabeb7366a96ee9fe1bbd7f90038))

- Moved autogluon training logic to methods
  ([`ec68800`](https://github.com/pmcdi/jarvais/commit/ec6880079b5d3232aadfa4681d628e961dd84f31))

- Moved survival model code to trainer
  ([`dcc298e`](https://github.com/pmcdi/jarvais/commit/dcc298eb2079d4bb2f980083cd38c1b1a26ef646))

- Os -> pathlib.Path
  ([`b8a35df`](https://github.com/pmcdi/jarvais/commit/b8a35df774531f38816208a1c163e8a16d72d0f5))

- Remove unused data
  ([`37edd52`](https://github.com/pmcdi/jarvais/commit/37edd5287a33f0ca35cb42341f2f3ea7c8ccd52d))

- Remove unused file
  ([`d270cf9`](https://github.com/pmcdi/jarvais/commit/d270cf9ffb474d5991ae97ec46faa9b05bb577f0))

- Remove unused from config
  ([`68d03e3`](https://github.com/pmcdi/jarvais/commit/68d03e3474a3d56c5b9114400b422350dc250f8b))

- Remvoe tutorials dir
  ([`5d751eb`](https://github.com/pmcdi/jarvais/commit/5d751ebb888214f335dd4b0ecd25c21e9934ead9))

- Spelling errors
  ([`00c4c08`](https://github.com/pmcdi/jarvais/commit/00c4c08e67fcdb391f7d4d11a8ca830f69fe95b7))

- Switch from os to pathlib in plot.py
  ([`2ea984b`](https://github.com/pmcdi/jarvais/commit/2ea984b21d979ab2a80882701785248062ff757d))

- Switch from os to pathlib in plot.py
  ([`8a113ee`](https://github.com/pmcdi/jarvais/commit/8a113ee721235b32bdb939d51144cbd678178cda))

- Undummify var
  ([`fd52ef7`](https://github.com/pmcdi/jarvais/commit/fd52ef7c0309e552464dd1bd8f78b25838740e93))
